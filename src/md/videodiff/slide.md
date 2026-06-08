---
marp: true
image: https://ogatakatsuya.github.io/slide/videodiff/slide.png
title: VideoDiff - Human-AI Video Co-Creation with Alternatives
description: CHI 2025 Paper Presentation
---
<!-- paginate: true -->

# VideoDiff

### Human-AI Video Co-Creation with Alternatives

CHI 2025 | Huh, Li, Pimmel, Shin, Pavel, Dontcheva (UT Austin & Adobe Research)

MIM Lab — Katsuya Ogata

<!-- 
この論文はCHI 2025（横浜）で発表されたもの。
著者はUT AustinとAdobe Researchの研究者6名。
今日は13分で背景・提案手法・評価・結果・Limitationまでカバーします。
-->

---
<!-- _header: Agenda -->
<!-- _class: agenda -->

1. Background: What Makes a Good Video Edit?
1. Problem: Comparing AI-Generated Alternatives
1. Proposed Method: VideoDiff
1. Evaluation
1. Results
1. Limitations & Future Work

---
<!-- _header: Background -->

## What Makes a Good Video Edit?

| Dimension | Key Question | Real Example of a Bad Edit |
|---|---|---|
| **① Story Coverage** | Are key moments included? | *"yolk separation step is not included"* |
| **② Visual Presentation** | Do B-rolls & text effects match narration? | *"Too many text effects"* |
| **③ Technical Quality** | Are there any edit errors? | *"mid-sentence jumps, flash frames"* |

<br>

> *"I cannot focus on multiple aspects at once while reviewing. When I'm checking the colors, I only look at colors."* — P5 (Professional Editor)

<!-- 
これが発表の根幹となるスライドです。

Formative Study（N=8のプロ編集者）で、複数の動画を比較しながらリアルタイムに書いたメモを分析。
3つの軸が浮かび上がりました：

① ストーリー・カバレッジ：何が含まれていて、何が欠けているか
  - 例：料理動画で「卵黄の分離工程」が抜けているかどうか
  - 情報の網羅性・重要シーンの有無

② 視覚的演出：BロールやテキストがナレーションやトーンにFitしているか
  - スタイルの一貫性（インフォグラフィック系 vs. 実写系）
  - エフェクトの頻度・タイミング

③ 技術的品質：AIが生成した動画に特有のエラー
  - ジャンプカット：シーンが不自然に飛ぶ
  - センテンス切断：文の途中で音声が途切れる
  - フラッシュフレーム：編集点に不意の1フレーム

P5の発言が示す通り、プロでも一度に複数観点を見ることはできない。
→ 何度も見直す必要があり、これが時間コストの根本原因
-->

---
<!-- _header: Background -->

## No Single "Correct" Edit Exists

<br>

| Context / Purpose | What "Good" Means |
|---|---|
| **Social Media Short** | Short length, strong opening hook |
| **Educational Video** | Full coverage of all topics, logical order |
| **Commercial / Ad** | Brand consistency, A/B test variants |
| **Documentary** | Long-form, narrative coherence |

<br>

### → Comparing **multiple alternatives** is essential to finding the best fit

<!-- 
良い編集の3軸を満たしていても、「どれを優先するか」はクライアントや用途で変わります。

プロへのインタビューより：
- P5：「クライアントは3つのバージョンを見たがるが、作るのに時間がかかりすぎる。
  だから小さな変更のバリエーションしか作らない。」
- P3・P4：「クライアントは複数バージョンの良い部分を組み合わせてほしいと言う」
- P5・P6：マーケティング会社向けにA/Bテスト用の2バージョンを作成

この「主観性」と「文脈依存性」こそが、AIが複数バリエーションを生成すべき根拠。
→ 複数バリエーションを提示することでクリエイターが最適解を選べる

そして、複数バリエーションがあるならば → 効率よく比較する手段が必要。
これがVideoDiffの出発点です。
-->

---
<!-- _header: Problem -->

## AI Can Now Generate Multiple Video Variations

<div style="display:flex; justify-content:center; align-items:center; gap:80px; margin-top:40px;">

<div style="text-align:center;">

![w:260](./image/capcut-seeklogo.png)

**CapCut**

</div>

<div style="text-align:center;">

![w:260](./image/opusclip.png)

**OpusClip**

</div>

</div>

<br>

### → One click generates **10+ different edited videos**

<!-- 
CapCutやOpusClipなどの現代のAIビデオ編集ツールは、
ワンクリックで10本以上の異なる編集バリエーションを自動生成できるようになった。

前のスライドで説明した通り、複数バリエーションを提示することは理想的。
しかしここで新たな問題が生じます。
-->

---
<!-- _header: Problem -->

## Comparing Videos Is Inherently Hard

| | Text / Image | **Video** |
|---|---|---|
| Comparison method | Side-by-side at a glance | **Must watch sequentially** |
| Spotting differences | Instant | **Requires full playback** |
| Comparing 10 candidates | Seconds | **Tens of minutes** |
| Cognitive load | Low | **High** |

<br>

> *"10 is a lot to compare at once. I need to take notes."* — P12

<!-- 
ビデオは「時間軸を持つメディア」なので、静止画やテキストと比較の方法が根本的に異なります。

8名のプロへのFormative Studyで判明した課題：
- 全員が「比較は非常に時間がかかる」と指摘
- 7名がメモを取りながら比較（内容の欠落・エラー・好き嫌いを記録）
- 5名が速度を上げてスキップしながら見るワークアラウンドを採用
- P3「文字起こしを読むだけでは不十分。文字がOKでも映像がおかしい場合がある」

さらに認知的に負荷が高い：
「ストーリー」「Bロール」「テキスト」「音楽」を同時に比較しなければならない。
P5「一度に複数の観点を見ることができない」→ 複数回の見直しが必要

P12「10本は多すぎてメモが必要」という発言がこの問題を象徴しています。
-->

---
<!-- _header: Background: Formative Study -->

## Formative Study with Professional Editors (N=8)

**Method:** Semi-structured interviews + comparison task (3 edited versions of same footage)

| Design Goal | Description |
|---|---|
| **D1** | Minimize redundant watching by **aligning** variations |
| **D2** | Support quick skimming by **highlighting differences** |
| **D3** | Enable **independent** comparison per editing stage |
| **D4** | Support comparison via **multiple modalities** (timeline, transcript, preview) |
| **D5** | Support **verification** of edit suggestions *(future work)* |
| **D6** | Support **management & customization** of variations |

<!-- 
Formative Studyの詳細：
- 参加者：Upworkで採用した8名のプロ（平均経験10.5年）
- コマーシャル・インタビュー・ドキュメンタリー・Vlog・ショートフォームなど多様なジャンル
- 補償：$28/時（自己申告の時給）、1時間のZoomセッション

比較タスクで提供した動画：
- F1：料理チュートリアル（1時間11分！）のAI編集版3本 vs 専門家編集版3本
- F2：TED Talk（12分）の同様の編集版

ここから6つの設計目標が導出されました。D5のみ今回は未実装（将来課題）。

特に重要なのはD4：単一モダリティ（文字起こしだけ、サムネイルだけ）では不十分。
P1「文字起こしだけでは映像の問題が見えない」
→ タイムライン・文字起こし・動画プレビューの3つを組み合わせる必要がある。
-->

---
<!-- _header: Proposed Method: VideoDiff -->

## VideoDiff — Overview

==![w:900](./image/teaser_new.png)=={.image}

---
<!-- _header: Proposed Method: VideoDiff -->

## System Overview

==![w:1150](./image/system_overview.jpg)=={.image}

<!-- 
VideoDiffはWebベースのツールで3つの編集タスクをサポートします：
1. Rough Cut（ラフカット）：ソース素材からキーシーンを選択
2. B-roll Insertion（Bロール挿入）：ナレーションを補完する映像を追加
3. Text Effects（テキストエフェクト）：ナレーションを強調するテキストを追加

インターフェースの主要コンポーネント：
- 左側：バリエーションリスト（ソート・フィルタ・ピン・アーカイブ機能付き）
- 中央：タイムラインビューまたは文字起こしビュー（切り替え可能）
- 右側：動画プレビュープレイヤー

技術実装：React.js + d3.js + Remotion（動画レンダリング）
AI：OpenAI Whisper（文字起こし）+ GPT-4o（編集提案・プロンプトエンジニアリング）
-->

---
<!-- _header: Proposed Method: VideoDiff -->

## Timeline View

==![w:1100](./image/timelines_new.jpg)=={.image}

<!-- 
タイムラインビューはD1・D2の設計目標を実現するコア機能です。

全バリエーションのタイムラインが同じ時間軸で縦に並びます（D1：アライメント）。
色付きセクションで「どのシーンが選ばれているか」が一目でわかります（D2：差分ハイライト）。

3段構成：
- 上段：Rough Cut（どのシーンが選択されているか）
- 中段：B-roll（いつどんな補足映像が入っているか）
- 下段：Text Effects（いつテキストが表示されるか）

「Edited View」（編集後）と「Source View」（元素材）を切り替え可能：
- Edited：編集後のタイムラインを表示
- Source：元素材の全体像を背景に、各バリエーションが選んでいる部分をハイライト

D3（独立比較）を実現：各編集ステージを分けて比較できる
-->

---
<!-- _header: Proposed Method: VideoDiff -->

### Transcript View — Side-by-Side Text Comparison (D2, D4)

==![h:560](./image/transcripts_new.jpg)=={.image}

<!-- 
文字起こしビューはD4（マルチモーダル比較）を実現するもう一つのコア機能です。

各バリエーションの文字起こしを横並びで表示（Side-by-Side）。
重要なキーワードをボールドでハイライト → スキミングが容易（D2を満たす）。

2種類のキーワード表示モード：
- Content Keywords：ナレーションの内容（話題・トピック）
- Edit Keywords：Bロールやテキストエフェクトの内容

P17の発言：
「タイムラインは動画の骨格を決めるのに使い、文字起こしは細部の確認に使う」

タイムラインとの使い分け：
- タイムライン：全体像・時間配分・視覚的なシーン構成
- 文字起こし：具体的な言葉・トピックの確認・音声内容の精査
→ 2つを組み合わせることでD4（マルチモーダル）が実現される
-->

---
<!-- _header: Proposed Method: VideoDiff -->

## Customization — Refine, Regenerate, Recombine

==![w:1050](./image/refine.jpg)=={.image}

<!-- 
VideoDiffはバリエーションを受動的に見るだけでなく、積極的に操作できます。

3つの操作（D6を実現）：
1. Regenerate：新しいテキストプロンプトで新しいバリエーションを生成
2. Refine：既存バリエーションを自然言語で修正
   - 例：「もっと短くして」「食料品についての部分だけ残して」
3. Recombine：複数バリエーションの良い部分を組み合わせる

整理機能：
- Pin：お気に入りを固定（後で参照しやすく）
- Archive：今は不要だが捨てたくないものを整理

各操作後にはAIが変更内容のサマリーを提示 → 何が変わったか把握しやすい

P21：「強化学習みたいに、使うにつれて好みを学習してほしい」（将来課題）
-->

---
<!-- _header: Evaluation -->

## User Study Design

| | Details |
|---|---|
| **Participants** | N=12 (Proficient: 6, avg. 8.3 yrs; Beginner: 6, avg. 3.8 yrs) |
| **Design** | Within-subjects (each participant tried both conditions) |
| **Conditions** | VideoDiff vs. Baseline (CapCut / OpusClip-style) |
| **Task 1: Comparison** | Answer 10 questions about 10 video variations (3-min limit/question) |
| **Task 2: Authoring** | Create a final video using the system |
| **Materials** | YouTube grocery haul videos (V1: 12:41, V2: 13:10) |
| **Order** | Counterbalanced assignment of conditions and videos |

<!-- 
実験設計の詳細を説明します。

参加者：
- Proficient（熟練者）6名：平均8.3年の経験（SD=2.94）
- Beginner（初心者）6名：平均3.8年の経験（SD=1.17）
- 補償：熟練者$75、初心者$30（1.5時間のZoomセッション）

ベースラインはCapCutやOpusClipに類似したUI。
10本の動画を一覧表示、ソート（長さ・エフェクト数）は可能だが、
差分の可視化・横断比較機能はなし。

比較タスクの10問：
- Single-selectとMulti-select混在
- ナレーション・映像・両方を確認する問題が混在
- 例：Q3「どの動画が最も話者のクローズアップが多いか？」（視覚）
- 例：Q4「どの動画がサラダレシピに言及しているか？」（音声・複数選択）

カウンターバランスで条件の順序と素材動画を制御。
-->

---
<!-- _header: Results -->

## Comparison Task: VideoDiff Halves Completion Time

==![w:820](./image/completion_time-1.png)=={.image}

> *"In only three minutes? I'll just have to guess as I cannot watch all these videos."* — P16 (Baseline)

<!-- 
比較タスクの定量的結果です。Wilcoxon符号順位検定を使用。

注目ポイント：
- 平均回答時間：74秒→38秒（約半分）
  - 特に「複数箇所を確認する問題」（Q6など）で大きな差
- 精度：ベースラインでは8名が3分制限内に正答を見つけられなかった
  - 5名は「推測」に切り替え（P16の発言が象徴的）

NASA-TLX（認知負荷）：
- Mental Demand・Effort・Frustration・Temporal Demandの全項目で有意に低い
- Performance（自己評価）は有意差なし

Usefulness評価（7点満点）：2.25→4.92と大幅に高い評価。

特筆：精度の差は視覚比較問題（Q3, Q6, Q7, Q9等）で特に顕著。
文字起こし検索で代替できた問題（Q1, Q4）は差が小さい。
-->

---
<!-- _header: Results -->

## Authoring Task: Higher Satisfaction & Creativity

==![w:1150](./image/stacked_bar-1.png)=={.image}

<!-- 
制作タスクの結果です。

重要な発見：
ベースラインでは「10個のバリエーションに圧倒される」と感じた参加者が4名。
しかしVideoDiffでは同じ10個でも圧倒感がなくなった。

P14の発言（ベースライン→VideoDiffの順で体験）：
「ベースラインの時は10個は多すぎると思ったが、VideoDiffだと差分が視覚化されているから
同じ10個でも全然多く感じなかった。」

Creativity Support Indexの全指標（探索・関与・効果感・表現力）でVideoDiffが優位。

探索的ケーススタディ（N=3、自分の素材持込み）：
- V3（物理学ポッドキャスト）：ニュートンの法則を全カバーするRough Cutを選び、Bロールを組み替え
- V4（アーティスト紹介）：「手動編集すると思考が固まる。バリエーションが新しいアイデアを生む」
- V5（IoTチュートリアル）：「ラフカットが普通2時間→5分に短縮！」

下図：探索的ケーススタディで使用された自前の素材動画（V3-V5）
-->

---
<!-- _header: Limitations & Future Work -->

| Category | Issue |
|---|---|
| **AI Hallucination** | AI claims edits were applied, but they are not reflected in the video |
| **Visual Fine Detail** | Short-duration objects missed by periodic filmstrip sampling (5 users failed Q5) |
| **Edit Verification (D5)** | Not implemented — detecting errors like jump cuts automatically |
| **Personalization** | System does not learn from user's selection history |
| **Long-form Video** | Not tested on documentaries or longer footage (1+ hour) |
| **Transcript-only Users** | Some users never switched from transcript view — modality preference varies |

<!-- 
Limitationsを正直に説明します。

1. AI幻覚問題（最も重大）：
   - P9が「フラッシュのある映像を除いて」とプロンプト → AIは除いたと応答、実際は残っていた
   - P9が「ホットスポット設定のシーンを追加して」→ 長すぎるバージョンが生成された
   - 現在のLLMの限界：動画の実際の内容を確認せずに自然言語で応答してしまう

2. 視覚的微細差分：
   - フィルムストリップは一定間隔でフレームを取得
   - 短時間だけ映るオブジェクト（例：はさみ）は見逃される
   - Q5（はさみが映っているか）で5名が誤答した主要因

3. D5（編集検証）は未実装：
   - ジャンプカットやセンテンス切断を自動検出する機能は将来課題

4. 個人化：
   - P21「強化学習みたいに好みを学習してほしい」
   - 現在は毎回ゼロから選ぶ

5. 長尺動画：
   - 使用した素材は12-13分。ドキュメンタリー（数時間）では別の課題が生じる可能性

ただし、これらのLimitationsはVideoDiffの比較・可視化機能の価値を否定するものではなく、
今後の改善方向を示すものです。
-->

---
<!-- _header: Conclusion -->

## Key Takeaways

**Problem:** AI generates multiple video alternatives, but comparing them is tedious and cognitively demanding

**Proposed:** VideoDiff — a human-AI co-creation tool with difference visualization

**Key Features:**
- Timeline view: aligned variations with color-coded sections
- Transcript view: side-by-side keyword-highlighted comparison
- Customization: Refine / Regenerate / Recombine

**Results (N=12, within-subjects):**

| | Baseline → VideoDiff |
|---|---|
| Comparison time | 74 sec → **38 sec** (×0.5) |
| Cognitive load | Significantly reduced (*p* < 0.05) |
| Video satisfaction | Significantly improved (*p* < 0.05) |

<!-- 
まとめです。

VideoDiffが解決した3つの核心課題：
1. 比較の効率化：同一時間軸のアライメント + 差分ハイライトで時間を半減
2. 認知負荷の軽減：多次元の視点（タイムライン・文字起こし・プレビュー）を組み合わせて整理
3. 創造性の支援：バリエーション探索でクリエイターが新しいアイデアを発見

より広い示唆：
AIが複数の提案を生成する時代において、「提案を比較するUI」の設計が重要になっている。
これはビデオだけでなく、文書・コード・デザインなど他のドメインにも応用可能な知見です。

VideoDiffは将来のHuman-AI共同制作における「比較中心設計」の先例となる研究です。

以上です。ご清聴ありがとうございました。
-->
