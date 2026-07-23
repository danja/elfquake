# ELFQuake

#### [English version](README.md)

#### [イタリア語版](README.it.md)

ELFQuake は、極低周波（ELF）・超低周波（VLF）の自然電波観測が、地震観測や天文・宇宙天気データを補完し、地震に関する有用な予測モデルの構築に役立つかを検証する研究プロジェクトです。

中心となる仮説は、自然の ELF/VLF 電波異常に、地震の発生履歴だけからは得られない情報が含まれる可能性があるというものです。研究として検証可能な機械学習手法と、再現可能な評価を重視しています。

現時点では地震予知能力を主張していません。主張を行うには、再現可能な地震のみのベースライン、時系列で分離したテスト期間、そしてマルチモーダル・アブレーションが必要です。

## Status

イタリアを対象とした、合成データから実データへの転移評価パイプラインが動作しています。現在の結果は歴史的な地震率ベースラインを上回っておらず、地震予測能力は確認されていません。

現在の重点は、時系列ホールドアウト評価、合成データの拡充、そして Cumiana VLF 観測の継続収集です。実データで十分な正例・負例が得られるまで、実データの教師あり学習は制限されています。詳細は [docs/report.md](docs/report.md)、[docs/model-comparison.md](docs/model-comparison.md)、[docs/next-actions.md](docs/next-actions.md) を参照してください。

## データとモデル

* INGV によるイタリアの地震イベント取得・正規化・時系列ウィンドウ作成。
* Cumiana VLF スペクトログラム画像の systemd による定期収集と特徴抽出。
* 天文・宇宙天気データの取得と正規化。
* CPU 上で動作する PyTorch の自己教師あり VLF 表現学習。
* 欠損モダリティ、シャッフル、除外を含む Transformer のアブレーション評価。
* 地震、VLF、天文データを扱うためのモジュール型マルチモーダル入力インターフェース。
* 山岳型サンドパイル・シミュレーションによる、地震様の雪崩イベントと piezo/VLF 類似信号の生成。
* 合成データで事前学習し、十分な実データが得られた後に実データで微調整するための転移学習用コード。

### 日本の ISEE VLF データ

日本の ISEE VLF/ELF データは、Moshiri 観測点の CDF スペクトログラムとして取得・正規化できます。現在はイタリアの本番データセットとは分離された比較用データです。

日本のデータおよびそこから作成した特徴量は **科学研究目的に限って使用** します。アーカイブの利用条件、原始 CDF、チェックサム、観測点メタデータを必ず保持してください。詳細は [docs/vlf-japan-isee.md](docs/vlf-japan-isee.md) を参照してください。

日本の CDF 取得・前処理を実行するには、次を使用します。

```sh
WINDOWS=data/derived/japan/japan.seismic_training_windows.csv \
  ./scripts/process-japan-vlf-manifest.sh
```

処理済みの複数 CDF ファイルを、重複しない地震ウィンドウ単位の特徴量にまとめるには、次を使用します。

```sh
WINDOWS=data/derived/japan/japan.seismic_training_windows.csv \
  ./scripts/build-japan-vlf-cdf-dataset.sh
```

## 自己教師あり学習

実 VLF ラベルが不足している間は、自己教師あり学習を既定の開発経路とします。これは、観測信号の一部を復元するなどの方法で表現を学習し、地震イベントの教師ラベルを直接仮定しない方法です。実データで正例と負例の両方が得られた後に、教師あり微調整を行います。

```sh
./scripts/pretrain-real-vlf-self-supervised.sh
./scripts/evaluate-self-supervised-transformer.sh
./scripts/score-real-vlf-anomaly-forecast.sh
```

## シミュレーション

シミュレーションは、山のような高さを持つ格子に、広域の荷重と固定された点状の応力源を繰り返し加えるモデルです。斜面が不安定になると、小さな雪崩が隣接セルへ高さを移します。

目的は、実際の地震データに十分近い構造を持つ合成時系列を作り、実データが少ない段階の深層学習モデルの訓練に利用できるかを調べることです。雪崩から得る直接的な地震様イベントと、piezo/VLF 類似センサー信号は別のモダリティとして保存されます。

```sh
./scripts/run-all.sh
```

これは地質学的モデルではなく、応力と解放の簡略化された類推です。合成データでの性能は、実際の地震予測能力を示しません。

## 評価方針

モデルは、まず単純な地震率ベースラインおよび地震のみのモデルと比較します。その後、VLF、天文、宇宙天気データを追加し、各モダリティを除外したアブレーションを行います。評価は時系列順に分割し、将来の期間を訓練に使用しません。

日本とイタリアのデータは、十分な重複期間とドメイン差の評価が得られるまで別々に評価します。日本データを加えた結果を、単純なデータ量の増加だけによるものと誤認しないためです。

## 主要ドキュメント

* [概要](docs/overview.md)
* [ドキュメント一覧](docs/README.md)
* [処理グラフ](docs/processing-graph.md)
* [次のアクション](docs/next-actions.md)
* [成功基準](docs/success-criteria.md)
* [開発環境](docs/development-environment.md)
* [データソース一覧](docs/source-inventory.md)
* [ISEE Japan VLF](docs/vlf-japan-isee.md)
* [モデル比較](docs/model-comparison.md)
* [サンドパイル・シミュレーション](docs/sandpile-simulation.md)
