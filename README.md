# 「おうちLLM　基礎を学んで作るおしゃべりチャットボット」サポートページ

Colorful Scoopの同人誌 「おうちLLM　基礎を学んで作るおしゃべりチャットボット」のサポートページです。

## 内容

### `src/*.py`

`src` 以下には、各章に対応したコードがあります。
例えば、`src/ch03_02.py` は第3章の2つ目のコードに対応しています。

### `data-shovel`

`data-shovel` 以下には、Colorful Scoopのキャラクター「シャベル」の会話データが入っています。

## 動作環境

モデルの学習コード `src/ch03_04.py` は GPU 環境で、それ以外は CPU 環境で実行しています。
GPU 環境は、GeForce RTX 3090 の GPU を使って実行しています。

### CPU環境

```sh
$ docker container run -w /work -v $(pwd):/work --rm -it python:3.8.10-buster bash
(container)$ apt update && apt install -y jq tree
(container)$ pip3 install -r requirements.txt
```

### GPU環境

```sh
$ docker container run --gpus all --ipc=host --rm -it -v $(pwd):/work -w /work nvidia/cuda:11.5.2-devel-ubuntu20.04 bash
(container)$ apt update && apt install -y python3 python3-pip jq tree
(container)$ pip3 install -r requirements.txt
```

## 実行方法

コードを `src/*.py` から指定して実行してください。

実行例

```sh
$ python3 src/ch01_01.py
```
