# AGENTS.md

## プロジェクト概要

**AivisSpeech Engine** は、[VOICEVOX ENGINE](https://github.com/VOICEVOX/voicevox_engine) をフォークして開発している日本語音声合成エンジンです。実体は FastAPI + Uvicorn による HTTP API サーバーで、デスクトップアプリ [AivisSpeech](https://github.com/Aivis-Project/AivisSpeech) に組み込まれて動作するほか、単独の API サーバーとしても利用できます。

- **音声合成モデル形式**: **AIVM** / **AIVMX** (`.aivmx`) という、学習済みモデル・ハイパーパラメータ・スタイルベクトル・話者メタデータを 1 ファイルにまとめた Aivis Project 独自のオープンフォーマットを採用しています。AIVM は Safetensors ベース、AIVMX は ONNX ベースで、AivisSpeech Engine は PyTorch 依存を排除しインストールサイズを抑えるため AIVMX のみをサポートします
- **モデルアーキテクチャ**: `Style-Bert-VITS2` / `Style-Bert-VITS2 (JP-Extra)` に対応
- **API 互換性**: VOICEVOX ENGINE の HTTP API と概ね互換性がありますが、`intonationScale` の意味変更・`tempoDynamicsScale` の追加・歌声合成系エンドポイントの `501 Not Implemented` 化など、独自の仕様変更が多数あります。詳細は README.md の「VOICEVOX API との互換性について」節を参照してください
- **Aivis Project エコシステムでの位置づけ**: `aivmlib` (AIVM/AIVMX メタデータの読み書きライブラリ)、[AivisHub](https://hub.aivis-project.com/) (音声合成モデルの共有ハブ、AivisSpeech Engine 起動時のデフォルトモデル配布・強制削除ルール配布元でもある)、Aivis Cloud API / Citoras (このリポジトリとは無関係な、GPU サーバー向け別製品) といった周辺プロジェクトが存在します。本リポジトリはあくまでローカル PC 上で 1 人で使うことを想定した設計であり、大量リクエストを捌く API サーバー用途には最適化されていません

### 技術スタック

- Python **3.11 固定** (`requires-python = ">=3.11,<3.12"` 、3.12 以降は非対応)
- パッケージ管理: **uv** (`pip` や `poetry` は使わない)
- Web フレームワーク: FastAPI (`fastapi-slim` ベース) + Uvicorn + Pydantic v2
- 音声合成: `style-bert-vits2` (tsukumijima によるフォーク、`pyproject.toml` の `[tool.uv.sources]` で git の特定 `rev` に固定) + ONNX Runtime
- 日本語処理: `pyopenjtalk-plus` 、`pyworld-prebuilt` 、`kanalizer` 、`e2k` (tsukumijima フォーク) 、`jaconv`
- エラーレポート: `sentry-sdk[fastapi]`
- リンタ/フォーマッタ: Ruff (lint + format) 、mypy (strict モード)
- ビルド: PyInstaller (`run.spec`)

### 主要コマンド

すべて `uv run task <name>` (taskipy) 経由で実行します。`pyproject.toml` の `[tool.taskipy.tasks]` に定義があります。

```bash
uv sync --group dev --group build  # 依存関係のインストール (開発時)
uv run task serve                  # 開発環境で AivisSpeech Engine を起動
uv run task lint                   # ruff check --fix + mypy による静的検査
uv run task format                 # ruff format によるコード整形
uv run task typos                  # typos によるタイポチェック
uv run task test                   # pytest 実行
uv run task update-snapshots       # テストスナップショットの更新
uv run task update-licenses        # 依存ライブラリのライセンス情報を再生成
uv run task build                  # PyInstaller によるビルド
```

## ディレクトリ構成

### トップレベル

- `run.py`: エントリーポイント。truststore の適用 (社内プロキシ環境での HTTPS 通信対策) 、Sentry の初期化、`AivmManager` / `StyleBertVITS2TTSEngine` などの組み立てとサーバー起動を行う
- `voicevox_engine/`: エンジン本体のソースコード (詳細は後述)
- `test/`: `unit/` (ユニットテスト) 、`e2e/` (API 単位の E2E テスト、`syrupy` によるスナップショットテストを含む) 、`benchmark/` (速度計測用スクリプト)
- `tools/`: ライセンス生成 (`generate_licenses.py`) 、Docker イメージ名生成、リリースビルド検証などの補助スクリプト
- `docs/`: VOICEVOX ENGINE 本家のドキュメントをそのまま引き継いだもの。AivisSpeech Engine 向けの更新は行われていない (後述の同期方針を参照)
- `engine_manifest.json` / `resources/`: エンジンマニフェスト (ブランド名・ポート番号・アイコンなど) と付随アセット
- `.github/workflows/`: CI 定義。`test.yml` は Windows / macOS (Intel・Apple Silicon) / Ubuntu の 4 環境で lint・format・mypy・pytest (coverage 計測込み) ・ライセンスチェックを実行する

### `voicevox_engine/` の構成

VOICEVOX ENGINE からのリブランディングを最小限に留める方針のため、ディレクトリ名は upstream のまま `voicevox_engine` を使い続けています (後述の同期方針を参照)。以下、AivisSpeech Engine で新規追加された固有部分と、upstream 由来で改変を受けている部分を区別して記載します。

- **`aivm_manager.py` / `aivm_infos_repository.py` (AivisSpeech Engine 固有)**: AIVMX モデルのインストール・スキャン・メタデータキャッシュ・AivisHub との同期を担う中核クラス群。VOICEVOX ENGINE における `MetasStore` の役割を代替しており、`metas/metas_store.py` 自体はコードとして残っているが無効化されている。`AivmManager` のコンストラクタでは、AivisHub から取得した「強制削除ルール」に一致する既存モデルの自動アンインストールと、「デフォルトモデル構成」に基づく自動インストール・自動更新を毎回行う
- **`app/`**: FastAPI アプリケーション定義
  - `routers/aivm_models.py`: AivisSpeech Engine 固有。AIVMX モデルの管理 API
  - `routers/character.py` , `routers/tts_pipeline.py`: upstream 由来だが、歌声合成系 API (`/singers` 、`/sing_frame_audio_query` など) やキャンセル可能音声合成 API を常に `501 Not Implemented` に、モーフィング API を常に `400 Bad Request` にする改変が入っている
- **`core/` (upstream 由来、実運用では不使用)**: `core_wrapper.py` / `core_adapter.py` / `core_initializer.py` は VOICEVOX CORE (共有ライブラリ) を読み込むための upstream コード。AivisSpeech Engine では `run.py` の `enable_mock: bool = True` の通り、常にモック実装 (`dev/core/mock.py`) のみが登録され、実際の音声合成処理はここを経由しない。upstream との差分を抑えるためコードごと残置されている
- **`tts_pipeline/`**: 音声合成パイプライン
  - `style_bert_vits2_tts_engine.py` (AivisSpeech Engine 固有、1000 行超): `AivmManager` が管理する AIVMX モデルを ONNX Runtime + Style-Bert-VITS2 で実際に音声合成する、本エンジンの心臓部
  - `tts_engine.py`: upstream 由来の `TTSEngine` / `TTSEngineManager` 型定義を維持しつつ、`run.py` 側で `make_tts_engines_from_cores()` の代わりに `StyleBertVITS2TTSEngine` を手動登録する形で差し替えて利用する
  - `kana_converter.py` , `katakana_english.py` , `njd_feature_processor.py` など: 読み・アクセント処理。upstream 由来だが、AivisSpeech Engine 向けの不具合修正 (辞書登録内容の反映など) が随時入る
- **`metas/` , `preset/` , `setting/` , `user_dict/` , `morphing/` , `library/`**: おおむね upstream 由来。`preset/model.py` や `model.py` の `AudioQuery` / `Mora` / `Preset` 型は、`intonationScale` の意味変更や `tempoDynamicsScale` 追加など AivisSpeech Engine 向けの仕様変更を受けている (詳細は README.md 参照)
- **`utility/`**: `aivishub_client.py` (AivisHub API クライアント、デフォルトモデル配布・強制削除ルール取得に使用) 、`sentry_utility.py` (Sentry 送信前フィルタ) 、`user_agent_utility.py` , `runtime_utility.py` などが AivisSpeech Engine 固有の追加分。`path_utility.py` などは upstream 由来
- **`model.py`**: `AudioQuery` などの中核データモデル。AivisSpeech Engine 向けの仕様変更とバリデーション強化が入っている (後述)
- **`engine_manifest.py` , `resource_manager.py` , `cancellable_engine.py`**: upstream 由来

## 直近の重要な設計判断・恒久的な制約

### onnxruntime のバージョン制約

`pyproject.toml` の onnxruntime 系依存は、OS ・アーキテクチャごとに `sys_platform` / `platform_machine` マーカーで細かく出し分けられており、それぞれに理由があります。迂闊にバージョン上限・下限を変更しないでください。

- **macOS x64 (Intel Mac)**: `onnxruntime==1.23.2` に固定。Intel Mac 向け wheel が提供される最終バージョンのため
- **macOS arm64 / Linux aarch64**: `onnxruntime>=1.24.0,<2.0.0`。1.23.2 に固定すると Intel Mac 側の制約に引きずられて universal lock が解決できなくなるため、明示的に 1.24.0 以降を指定している (コミット `e724f0d`) 。当初は「Intel Mac 向け wheel の存在を lock 時に必須化する」ための `required-environments` 設定も入れていたが、実際に依存解決を検証した結果不要と判断し削除した (コミット `ef581ea`)
- **Linux x64 (GPU 版、`onnxruntime-gpu`)**: `>=1.24.0,<1.27.0` に固定。1.27.0 以降は import した時点で CUDA 13 系の共有ライブラリが存在しないとエラーになる仕様変更が入ってしまい、CUDA 13 未導入の環境で起動不能になる。そのため当面 1.26.x 系に固定している (コミット `d69f322`) 。upstream (onnxruntime-gpu) 側の互換性問題であり、AivisSpeech Engine 側のバグではない

### Sentry エラーレポートの運用

- 旧バージョンに埋め込まれた DSN 宛にエラー報告が大量に届き続ける「エラー報告爆撃」状態が発生したため、旧 DSN を無効化し新しい DSN に切り替えた (コミット `a0db92b`) 。DSN を変更しても過去バージョンの利用者からの送信は止まらないため、今後もこの種の対応が必要になる可能性がある
- `voicevox_engine/utility/sentry_utility.py` の `filter_sentry_event()` を `sentry_sdk.init()` の `before_send` に登録し、ユーザー環境に依存する復旧不能な既知エラー (ポート使用中・辞書一時ファイル生成失敗・メモリ/GPU リソース枯渇・AIVM モデル未インストール・AudioQuery バリデーションエラーなど) を送信前に除外している (コミット `7767b40`) 。Sentry 側で受信後にフィルタしてもクォータを消費するため、SDK 側 (クライアント) で止める設計になっている
- 同コミットで `traces_sample_rate` を `0.0` に変更し、トレース・プロファイリングは無効化している。ローカルで動くアプリのため、エラー以外の利用状況までは収集しない方針

### VOICEVOX CORE は常にモック

`run.py` の `enable_mock: bool = True` (コメント「常にモック版 VOICEVOX CORE を利用する」) の通り、AivisSpeech Engine では VOICEVOX CORE 本体の共有ライブラリを読み込みません。実際の音声合成は `AivmManager` が読み込んだ AIVMX モデルを `StyleBertVITS2TTSEngine` が処理する経路のみが使われます。`core/core_wrapper.py` などの VOICEVOX CORE 統合コードや `make_tts_engines_from_cores()` は upstream との差分を抑えるために残置されているだけで、`run.py` 内でも該当の呼び出しはコメントアウトされており実行パスに乗りません。

### AudioQuery のバリデーション強化

`vowel` フィールドに音素以外の値を入れるなど、本来受け付けられない入力によってコア内部でクラッシュする事例が Sentry 上で観測されたため、`voicevox_engine/model.py` の Pydantic バリデーションを強化し、入力バリデーションの時点でエラーを返すよう修正した (コミット `956ec5a`) 。コア内部でのクラッシュはプロセス全体に影響するため、境界であるモデル層で弾く方針を取っている。

## upstream (VOICEVOX ENGINE) との同期方針

README.md の「開発方針」節に明記されている、以下の方針を必ず踏まえてください。詳細や理由は README.md 側も参照してください。

- **改変は必要最小限に留める**: VOICEVOX 最新版への追従を容易にするため。`voicevox_engine` ディレクトリのリネームなど、import 文の差分が膨大になるリブランディングは行わない
- **リファクタリングは行わない**: upstream とのコンフリクトが発生しやすく、また upstream のコード全体に精通しているわけではないため
- **AivisSpeech で使わない機能もコードごと削除しない**: 歌声合成機能などが該当する。無効化する場合はコメントアウトで対応し (`#` で大量になる場合は `""" """` を使う) 、削除はしない。ただし Dockerfile や GitHub Actions などの構成ファイル・ビルドツール類はこの限りではない (元々 AivisSpeech Engine 側での改変量が大きいため)
- **ドキュメントの更新は行わない**: 保守や追従が困難なため。`docs/` 以下は upstream のまま放置されている
- **テストコードの追加は行わない**: AivisSpeech Engine 向けの改変にともなうテスト保守コストを避けるため。既存のテストのみ、通るように修正・コメントアウトで消極的に維持する (AivisSpeech Engine の改変によりテストスナップショットは upstream と異なる)

この方針の帰結として、コード中に一見 "死んでいる" ように見える upstream 由来のコードやコメントアウトされた大きなブロックが多数存在しますが、これは意図的な設計判断であり、勝手に削除・リファクタリングしないでください。

## 開発時の注意

- **パッケージ管理は uv に固定**: `python` / `pip` を直接使わず、必ず `uv run` 経由でコマンドを実行する
- **`style-bert-vits2` / `e2k` は git 依存**: `pyproject.toml` の `[tool.uv.sources]` で tsukumijima のフォークリポジトリの特定コミット (`rev`) に固定されている。更新する際は `rev` を明示的に張り替える必要があり、`uv.lock` も連動して更新される
- **リンタ/フォーマッタ/型チェック**: 編集後は必ず `uv run task lint` (ruff check --fix + mypy strict) と `uv run task format` (ruff format) を実行する。CI (`test.yml`) でも同じチェックが Windows / macOS / Ubuntu の 4 環境で走る
- **pre-commit は push 時のみ実行**: `.pre-commit-config.yaml` の各フックはすべて `stages: [pre-push]` であり、コミット時には走らない。コミット前に手元で `uv run task lint` などを明示的に実行しておくこと
- **テスト**: `uv run task test` (pytest) 。スナップショット更新は `uv run task update-snapshots` (syrupy) 。`uv.lock` の整合性チェックは CI で `uv lock --check` として実行される
- **ライセンス情報**: `uv run task update-licenses` で `resources/engine_manifest_assets/dependency_licenses.json` を再生成する。`pip-licenses` は `5.0.0` に厳密固定 (`pyproject.toml` の NOTE 参照、issue #1281 対応のため)
- **Ruff の pydocstyle 設定**: `convention = "numpy"` 、かつ `D200` / `D202` / `D205` / `D400` / `D403` を無効化している。特に `D400` (末尾ピリオド必須) は日本語の句点「。」に対応していないための無効化であり、日本語 Docstring の文体に合わせた設定になっている
