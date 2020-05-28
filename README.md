# Nishika 財務・非財務情報を活用した株主価値予測 2位ソースコード

https://www.nishika.com/competitions/4/summary

## Approach

1. Create 1180 features only from financial information.
2. Make 3 predictions by LightGBM and CatBoost with 5 cross-validation.
3. Netflix blending.

For more details, see [approach.md](docs/approach.md).

## Input Data

I only use `{2014-2017}/documents.csv`.

```
input
└── data
    |── 2014
    |   └── documents.csv
    |── 2015
    |   └── documents.csv
    |── 2016
    |   └── documents.csv
    └── 2017
        └── documents.csv
```

## Environment

```bash
docker-compose up --build -d
docker exec -it nishika-yuho bash
```

## Run

```
sh run.sh
```
