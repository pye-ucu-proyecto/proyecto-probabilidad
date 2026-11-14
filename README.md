# Caos determinista y pseudoaleatoriedad con el mapa logístico

Proyecto de investigación — Probabilidad y Estadística Aplicada, Universidad Católica del Uruguay.

## Requisitos
- Python 3.12

## Ejecución
```bash
pip install -r requirements.txt
python main.py
```

Tarda menos de un minuto. Al terminar se crea:

- `salida/figuras/` — las 12 figuras del informe (fig1 a fig12)
- `salida/resultados.json` — todos los valores numéricos de las tablas

El programa no usa números aleatorios: todas las semillas son deterministas,
por lo que cada ejecución produce exactamente los mismos resultados.
