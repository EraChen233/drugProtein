# 项目说明
```json
drugProtein
├── pydpi                "利用pydpi下载数据"
│   ├── code             "下载数据代码、格式化等等"
│   ├── example.py       "pydpi提供的例子文件"
│   ├── fetchdpi.py      "下载dpi1、dpi2数据入口"
│   ├── fetchDrug.py     "单独下载的药物数据入口"
│   ├── fetchProtein.py  "单独下载蛋白质数据入口"
│   └── ...
├── xgboost              "使用xgboost进行药物靶标预测"
│   ├── code             "进行特征工程等数据出来工作"
│   │    ├── gbdt.py     "用xgboost进行训练"
│   │    ├── gbdt_lr.py  "用xgboost进行训练，然后用xgboost生成的树作为lr模型的特征输入"
│   │    └── ...
│   ├── run.py           "xgboost训练入口，调用code文件夹的gbdt或者gbdt_lr进行训练"
│   └── ...
└── readme.md
```

