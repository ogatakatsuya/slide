1. 新しいスライドを作成する

```sh
./scripts/new-slide.sh {slide_title}
```

2. 開発用のサーバーをたちげる

```sh
pnpm dev
```

3. ビルドを行う

```sh
pnpm build
```

## PDF → PNG 変換

スライドに論文のPDF図を使いたい場合は `pdftoppm`（poppler）で変換する。

```bash
# brew install poppler
pdftoppm -png -r 150 path/to/figure.pdf output_prefix
# → output_prefix-1.png として出力される
```

`-r 150` は解像度（DPI）。スライド用は 150 で十分、高品質にしたい場合は 300。

## PDF 読み取り

講義資料のPDFは `pdftotext` でテキスト抽出してから読む（20MB超のファイルは直接 Read ツールでは読めないため）。

```bash
# テキスト全体を標準出力へ
pdftotext "path/to/file.pdf" -

# 量が多い場合は分割して取得
pdftotext "path/to/file.pdf" - | sed -n '1,300p'
pdftotext "path/to/file.pdf" - | sed -n '300,600p'
```