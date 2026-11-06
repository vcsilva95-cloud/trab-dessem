# Programacao semanal hidrotermica

Aplicacao Streamlit para otimizacao semanal hidrotermica com PySCIPOpt.

## Arquivos
- `app.py`: interface Streamlit
- `modelo.py`: modelo de otimizacao
- `VERAO.csv` e `INVERNO.csv`: perfis semanais de carga
- `requirements.txt`: dependencias para Streamlit Community Cloud

## Execucao local
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Streamlit Community Cloud
Use `app.py` como *Main file path*. Se disponivel em **Advanced settings**, selecione Python 3.15, que corresponde ao ambiente local utilizado no desenvolvimento.
