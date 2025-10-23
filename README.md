# Ironia-MVP: Detecção de Ironia em Redes Sociais (PT-BR)

Este projeto implementa um pipeline de detecção de ironia em textos de redes sociais em português brasileiro, utilizando modelos de aprendizado de máquina clássicos (SVM) e modelos transformadores avançados (BERTimbau).

---

## Estrutura de Entrada

O arquivo CSV deve conter ao menos:

| Coluna | Descrição |
|--------|-----------|
| `text` | Conteúdo textual do post |
| `label` | 1 = irônico, 0 = não-irônico |

Opcionalmente:

| Coluna | Papel |
|--------|------|
| `author_id` | Controlar vazamento de usuário entre train/val/test |
| `timestamp` | Permitir split temporal sem mistura de épocas |

---
## Dependências

numpy	- array e cálculos numéricos
pandas	- leitura do CSV e manipulação de dados
scikit-learn	- SVM, TF-IDF, métricas, splits
scipy	- hstack de matrizes esparsas
ftfy	- correção de encoding
emoji	- detecção/substituição de emojis
unidecode	- suporte opcional ao texto
transformers	- modelo BERTimbau e Trainer
datasets	- DatasetDict do HuggingFace
evaluate	- métricas HF (f1, accuracy)
accelerate	- otimizações de treino
torch	- rede neural do BERT

## Como Executar

Crie e ative um ambiente virtual (opcional, mas recomendado):

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
````

Para instalar as dependências
````bash
pip install numpy pandas scikit-learn scipy ftfy emoji unidecode
pip install transformers datasets evaluate accelerate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
````
Para rodar o código, é necessário dar uma base de dados como input. Já temos uma de exemplo aqui neste repositório

Para rodar somente o SVM
````bash
python ironia_mvp.py --data tweets.csv --do_svm
````

Para rodar somente o modelo BERTimbau
````bash
python ironia_mvp.py --data tweets.csv --do_bert
````

Para rodar ambos os modelos
````bash
python ironia_mvp.py --data tweets.csv --do_svm --do_bert
````

##Parâmetros Opcionais

--no_group_user	- Ignora agrupamento por autor mesmo se houver author_id
--group_time	- Faz split temporal se houver timestamp
--epochs	- Número de épocas para o BERT
--max_len	- Tamanho máximo dos tokens
--lr	- Taxa de aprendizado
--batch	- Tamanho do batch


##📊 Saídas e Métricas

Ao final da execução são exibidos:
📈 F1-macro por classe
📉 Matriz de confusão
📌 Relatório completo (precision / recall / F1)
📦 Melhor modelo BERT é salvo em out_bert/



##Licenciamento

✅ Licença MIT 
