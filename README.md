# DL基礎講座2024　最終課題「脳波分類」

## 環境構築

```bash
conda create -n dl-meg python=3.10
conda activate dl-meg
pip install poetry
poetry install
```

## モデルを動かす

### 訓練

```bash
poetry run python main.py

# オンラインで結果の可視化（wandbのアカウントが必要）
poetry run python main.py use_wandb=True
```

- `outputs/{実行日時}/`に重み`model_best.pt`と`model_last.pt`，テスト入力に対する予測`submission.npy`が保存されます．`submission.npy`をOmnicampusに提出することで，test top-10 accuracyが確認できます．

  - `model_best.pt`はvalidation top-10 accuracyで評価

- 訓練時に読み込む`config.yaml`ファイルは`train.py`，`run()`の`@hydra.main`デコレータで指定しています．新しいyamlファイルを作った際は書き換えてください．

### 評価のみ実行

- テストデータに対する評価のみあとで実行する場合．出力される`submission.npy`は訓練で最後に出力されるものと同じです．

```bash
poetry run python eval.py model_path={評価したい重みのパス}.pt
```

## データセット[[link](https://openneuro.org/datasets/ds004212/versions/2.0.0)]の詳細

- 1,854クラス，22,448枚の画像（1クラスあたり12枚程度）
  - クラスの例: airplane, aligator, apple, ...

- 各クラスについて，画像を約6:2:2の割合で訓練，検証，テストに分割

- 4人の被験者が存在し，どの被験者からのサンプルかは訓練に利用可能な情報として与えられる (`*_subject_idxs.pt`)．

### データセットのダウンロード

- [こちら](https://drive.google.com/drive/folders/1pgfVamCtmorUJTQejJpF8GhvwXa67rB9?usp=sharing)から`data.zip`をダウンロードし，`data/`ディレクトリに展開してください．

- 画像を事前学習などに用いる場合は，ドライブから`images.zip`をダウンロードし，任意のディレクトリで展開します．{train, val}_image_paths.txtのパスを使用し，自身でデータローダーなどを作成してください．

## タスクの詳細

- 本コンペでは，**被験者が画像を見ているときの脳波から，その画像がどのクラスに属するか**を分類します．

- 評価はtop-10 accuracyで行います．
  - モデルの予測確率トップ10に正解クラスが含まれているかどうか
  - つまりchance levelは10 / 1,854 ≒ 0.54%となります．
