# 分類ルール

## 破棄するもの

AivisSpeech Engine のデバッグに使えないと実データから確認できた署名だけを破棄します。

既知の破棄分類は以下です。

- クライアント切断: `ConnectionResetError`、`[WinError 10054]`、h11 `MUST_CLOSE`
- ポート競合: `address already in use`、Windows 日本語 / 韓国語のソケットアドレスエラー、bind 時のアクセス拒否
- ユーザー辞書の一時ファイル: `user.dict_compiled-*.tmp` の作成失敗・欠損
- メモリ・GPU リソース不足: `bad allocation`、`Failed to allocate memory`、`Not enough memory resources`、`out of memory`、単語境界付き `OOM`、`MemoryError`、DirectML デバイス停止、CUDA / cuDNN 初期化失敗
- インストール済みリソース欠損: 話者・モデル・スタイル欠損、デフォルトモデル削除、最後のモデル削除
- API バリデーションで扱うべき不正リクエスト: `AudioQuery`、`Setting`、`ParseKanaBadRequest`、巨大整数変換、正規化後テキスト長すぎ
- ローカルモデル・ユーザーデータ破損: 不正な AIVM / AIVMX、壊れたモデルファイル、pickle 読み込み失敗、manifest / ユーザー辞書のバリデーションエラー
- ローカル BERT キャッシュ・Hugging Face キャッシュ欠損: BERT ONNX / tokenizer / config ファイル欠損、ローカルキャッシュ未検出
- 外部ダウンロード・プロキシ・SSL・外部サービス失敗: AIVMX ダウンロード失敗、プロキシ 407、`ConnectTimeout`、`ReadTimeout`、`ReadError`、AIVMX ダウンロード URL の `HTTPStatusError`、`RemoteProtocolError`、`ChunkedEncodingError`、`IncompleteRead`、Hugging Face 503 / XetHub URL
- 中断・終了処理: `KeyboardInterrupt`、`CancelledError`、Uvicorn 終了時 traceback
- 壊れた実行環境: `Permission denied: 'dmesg'`、Windows コードページ由来の ONNX 初期化失敗、壊れた `regex` インストール
- 壊れたローカルインストール: `resources/engine_manifest_assets/icon.png` 欠損

## 残すもの

AivisSpeech Engine、SBV2、g2p、tokenizer、テキスト前処理の不具合を示す可能性がある issue は残します。

既知の残す分類は以下です。

- `ValueError: Input must be katakana only`
- `Unexpected phone`
- `InvalidPhoneError`
- g2p または `kata_tone2phone_tone` 内の文字化け風 `KeyError`
- synthesis / audio_query 内の `UnicodeDecodeError`（event detail でローカルファイル破損だと確認できた場合を除く）
- `adjust_word2ph` 内の `AssertionError`
- ONNX の invalid rank、index bounds、negative dimensions、unknown errors（イベント詳細でメモリ・GPU 不足だと確認できた場合を除く）
- テキスト処理由来の Sudachi 辞書 / plugin 構築エラー

これらは件数が多くても、プロジェクト方針としては残します。  
強化済みバリデーション後の実データで「特定署名がユーザー入力ノイズ」と確認できた場合だけ、個別に破棄分類へ移します。

## 修正対象の扱い

残す issue は、さらに修正先で分けます。

- AivisSpeech Engine 内で完結する場合: このリポジトリで修正し、絞り込みテストまたは既存テストを追加・更新します
- 入力検証で止められる場合: Engine 側の Pydantic モデル、API 境界、前処理入口で拒否する実装を優先します
- SBV2 コア側の実装が必要な場合: このリポジトリだけで直せるように見せず、スタック・入力・再現条件・修正候補をユーザーへ報告します
- 依存ライブラリ側の問題が濃い場合: upstream issue、バージョン固定、回避策の有無を確認し、Engine 側で安全に回避できる場合だけ実装します
- 辞書データ・モデル配布物の問題が濃い場合: どのデータが壊れているか、どの配布経路の更新が必要かをユーザーへ報告します

修正先を判断できない場合は、Sentry のタイトルだけで決めません。  
イベント詳細、スタック、リクエスト入力、現在の Engine 実装、SBV2 側の該当実装を確認してから、実装するか報告に留めるかを決めます。

## 保留するもの

タイトルだけを根拠に、以下の汎用例外へフィルタを追加しません。

- 汎用的な `TypeError`
- 汎用的な `OSError`
- 汎用的な `ValueError`
- 汎用的な `HTTPException`
- 汎用的な `RuntimeError`

先にイベント詳細、スタック、リクエスト context を確認します。  
それでも確定できない場合は、黙ってフィルタせず、残した理由を報告します。
