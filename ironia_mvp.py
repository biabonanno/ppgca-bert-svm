import argparse, re, random, math
import numpy as np
import pandas as pd
from typing import List, Dict, Any

# ---------- util ----------
SEED = 42
random.seed(SEED); np.random.seed(SEED)

def set_seed(seed=SEED):
    import torch
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)

# ---------- pré-processamento p/ social PT-BR ----------
import emoji as emoji_lib
from ftfy import fix_text

EMOJI_TOKEN = "<EMOJI>"
URL_TOKEN   = "<URL>"
USER_TOKEN  = "<USER>"
HASHTAG_TOKEN = "<HASHTAG>"

URL_RE   = re.compile(r"https?://\S+|www\.\S+")
USER_RE  = re.compile(r"(?<!\w)@[\w_]+")
HASH_RE  = re.compile(r"(?<!\w)#[\w_]+")
LAUGH_RE = re.compile(r"(k|rs){3,}", flags=re.IGNORECASE)
ELONG_RE = re.compile(r"(\w)\1{2,}", flags=re.IGNORECASE)  # looongo -> loo
MULTI_PUNCT_RE = re.compile(r"([!?.]){2,}")

def normalize_social(text: str) -> str:
    t = fix_text(text)  # conserta encoding bizarro
    # compatível com emoji>=2.x (callable recebe (emoji, data))
    t = emoji_lib.replace_emoji(t, replace=lambda e, data=None: f" {EMOJI_TOKEN} ")
    t = URL_RE.sub(f" {URL_TOKEN} ", t)
    t = USER_RE.sub(f" {USER_TOKEN} ", t)
    # manter a hashtag como sinal
    t = HASH_RE.sub(lambda m: f" {HASHTAG_TOKEN}:{m.group(0)[1:]} ", t)
    t = LAUGH_RE.sub(" risos ", t)
    t = ELONG_RE.sub(r"\1\1", t)          # reduz alongamentos, mas mantém 2
    t = MULTI_PUNCT_RE.sub(r"\1\1", t)    # "!!!" -> "!!"
    t = re.sub(r"\s+", " ", t).strip()
    return t

# ---------- carga e splits ----------
from sklearn.model_selection import train_test_split, GroupShuffleSplit

def load_and_split(csv_path: str, test_size=0.2, val_size=0.1, group_by_user=True, group_by_time=False):
    df = pd.read_csv(csv_path)
    assert {"text", "label"}.issubset(df.columns), "CSV precisa das colunas text,label"
    # normaliza
    df["text_norm"] = df["text"].astype(str).apply(normalize_social)
    df["label"] = df["label"].astype(int)

    # grupos
    if group_by_user and "author_id" in df.columns:
        groups = df["author_id"].astype(str)
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
        train_idx, test_idx = next(gss.split(df, groups=groups))
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_test  = df.iloc[test_idx].reset_index(drop=True)
        # validação a partir do train
        groups_tr = df_train["author_id"].astype(str)
        gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size/(1-test_size), random_state=SEED)
        tr_idx, va_idx = next(gss2.split(df_train, groups=groups_tr))
        df_tr = df_train.iloc[tr_idx].reset_index(drop=True)
        df_va = df_train.iloc[va_idx].reset_index(drop=True)
    elif group_by_time and "timestamp" in df.columns:
        # split por tempo (ordenado)
        df = df.sort_values("timestamp")
        n = len(df)
        n_test = int(math.ceil(n*test_size))
        n_val  = int(math.ceil(n*val_size))
        df_tr = df.iloc[: n - n_test - n_val]
        df_va = df.iloc[n - n_test - n_val : n - n_test]
        df_test = df.iloc[n - n_test :]
        df_tr, df_va, df_test = df_tr.reset_index(drop=True), df_va.reset_index(drop=True), df_test.reset_index(drop=True)
    else:
        df_tr, df_tmp = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=SEED)
        df_tr, df_va  = train_test_split(df_tr, test_size=val_size/(1-test_size), stratify=df_tr["label"], random_state=SEED)
        df_tr, df_va, df_test = map(lambda x: x.reset_index(drop=True), [df_tr, df_va, df_tmp])

    return df_tr, df_va, df_test

# ---------- métricas ----------
from sklearn.metrics import classification_report, f1_score, confusion_matrix

def eval_and_print(y_true, y_pred, title=""):
    print("\n" + "="*80)
    print(title)
    print("-"*80)
    print(classification_report(y_true, y_pred, digits=3))
    print("F1-macro:", f1_score(y_true, y_pred, average="macro"))
    print("Matriz de confusão:\n", confusion_matrix(y_true, y_pred))

# ---------- baseline SVM ----------
def run_svm(df_tr, df_va, df_test):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.base import BaseEstimator, TransformerMixin
    from scipy.sparse import hstack

    # features de sinais (emoji/hashtag/pontuação etc.)
    class SignalFeats(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            out = []
            for t in X:
                em = t.count(EMOJI_TOKEN)
                hs = len(re.findall(rf"{HASHTAG_TOKEN}:", t))
                ex = t.count("!!")
                qn = t.count("??")
                qt = t.count('"') + t.count("'")
                flip = int(any(w in t.lower() for w in ["sqn", "só que não", "aham", "imagina"]))
                out.append([em, hs, ex, qn, qt, flip])
            return np.array(out, dtype=np.float32)

    # função auxiliar com fallback
    def fit_tfidf_with_fallback(texts):
        tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,          # antes era 3
            max_df=0.99,       # antes era 0.90
            sublinear_tf=True,
            strip_accents=None,
            token_pattern=r"(?u)\b\w+\b|<\w+(?::\w+)?>"
        )
        try:
            return tfidf, tfidf.fit_transform(texts)
        except ValueError:
            # fallback agressivo p/ bases muito pequenas
            tfidf.set_params(min_df=1, max_df=1.0)
            return tfidf, tfidf.fit_transform(texts)

    # prepara os dados
    Xtr = [t for t in df_tr["text_norm"].tolist() if t.strip()]
    Ytr = df_tr["label"].values
    Xva = df_va["text_norm"].tolist(); Yva = df_va["label"].values
    Xte = df_test["text_norm"].tolist(); Yte = df_test["label"].values

    # vetoriza com fallback
    tfidf, Xtr_tfidf = fit_tfidf_with_fallback(Xtr)
    Xva_tfidf = tfidf.transform(Xva)
    Xte_tfidf = tfidf.transform(Xte)

    # adiciona features de sinais
    sig = SignalFeats()
    Xtr_sig = sig.fit_transform(Xtr)
    Xva_sig = sig.transform(Xva)
    Xte_sig = sig.transform(Xte)

    Xtr_all = hstack([Xtr_tfidf, Xtr_sig])
    Xva_all = hstack([Xva_tfidf, Xva_sig])
    Xte_all = hstack([Xte_tfidf, Xte_sig])

    # SVM linear
    clf = LinearSVC(class_weight="balanced", random_state=SEED)
    clf.fit(Xtr_all, Ytr)

    # validação
    yva = clf.predict(Xva_all)
    eval_and_print(Yva, yva, "SVM (val)")

    # teste
    yte = clf.predict(Xte_all)
    eval_and_print(Yte, yte, "SVM (test)")

# ---------- BERTimbau ----------
def run_bert(df_tr, df_va, df_test, model_name="neuralmind/bert-base-portuguese-cased", max_len=160, epochs=3, lr=2e-5, batch=16):
    set_seed(SEED)
    import torch
    from datasets import Dataset, DatasetDict
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
    from transformers import TrainingArguments, Trainer
    import evaluate as hf_evaluate
    from sklearn.metrics import classification_report, confusion_matrix

    # dataset HF
    ds = DatasetDict({
        "train": Dataset.from_pandas(df_tr[["text_norm","label"]].rename(columns={"text_norm":"text"})),
        "validation": Dataset.from_pandas(df_va[["text_norm","label"]].rename(columns={"text_norm":"text"})),
        "test": Dataset.from_pandas(df_test[["text_norm","label"]].rename(columns={"text_norm":"text"})),
    })

    tok = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=max_len)

    ds_tok = ds.map(tokenize, batched=True, remove_columns=["text"])
    collator = DataCollatorWithPadding(tokenizer=tok)

    num_labels = 2
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # pesos por classe (desbalanceamento)
    class_counts = np.bincount(df_tr["label"].values, minlength=num_labels)
    class_weights = class_counts.sum() / (num_labels * class_counts + 1e-9)
    cw_tensor = torch.tensor(class_weights, dtype=torch.float)

    metric_f1 = hf_evaluate.load("f1")
    metric_acc = hf_evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": metric_acc.compute(predictions=preds, references=labels)["accuracy"],
            "f1_macro": metric_f1.compute(predictions=preds, references=labels, average="macro")["f1"],
        }

    # treinamento
    args = TrainingArguments(
        output_dir="out_bert",
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        seed=SEED,
        fp16=False,
        report_to="none"
    )

    # loss com class weights
    def custom_loss(model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=cw_tensor.to(logits.device))
        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            return custom_loss(model, inputs, return_outputs)

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    # avaliação final
    eval_val = trainer.evaluate(ds_tok["validation"])
    print("\n" + "="*80)
    print("BERTimbau (val):", eval_val)

    preds = trainer.predict(ds_tok["test"])
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=-1)
    print("\n" + "="*80)
    print("BERTimbau (test)")
    print(classification_report(y_true, y_pred, digits=3))
    print("Matriz de confusão:\n", confusion_matrix(y_true, y_pred))

# ---------- main ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="caminho para CSV com colunas text,label,[author_id,timestamp]")
    ap.add_argument("--no_group_user", action="store_true", help="não fazer split por usuário mesmo se author_id existir")
    ap.add_argument("--group_time", action="store_true", help="split por tempo se timestamp existir")
    ap.add_argument("--do_svm", action="store_true")
    ap.add_argument("--do_bert", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max_len", type=int, default=160)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    df_tr, df_va, df_te = load_and_split(
        args.data,
        group_by_user=not args.no_group_user,
        group_by_time=args.group_time
    )

    print(f"train={len(df_tr)} val={len(df_va)} test={len(df_te)}")
    print("positivos train/val/test:", df_tr['label'].mean(), df_va['label'].mean(), df_te['label'].mean())

    if args.do_svm:
        run_svm(df_tr, df_va, df_te)

    if args.do_bert:
        run_bert(df_tr, df_va, df_te, max_len=args.max_len, epochs=args.epochs, lr=args.lr, batch=args.batch)
