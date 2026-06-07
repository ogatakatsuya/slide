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

## PDF 読み取り

講義資料のPDFは `pdftotext` でテキスト抽出してから読む（20MB超のファイルは直接 Read ツールでは読めないため）。

```bash
# テキスト全体を標準出力へ
pdftotext "path/to/file.pdf" -

# 量が多い場合は分割して取得
pdftotext "path/to/file.pdf" - | sed -n '1,300p'
pdftotext "path/to/file.pdf" - | sed -n '300,600p'
```