# フィルタ実装

## コード方針

`voicevox_engine/utility/sentry_utility.py` を編集します。

広い例外型チェックより、署名ベースの正規表現グループを優先します。  
目的は、エンジン側の不具合候補を残しつつ、既知ノイズだけを破棄することです。

パスを照合するときは、`C:\Users\user` のようなユーザーホーム固定パスを直書きしません。  
環境ごとに変わる部分はワイルドカードにし、複数レイアウトで発生しうるエラーでは Windows / POSIX の両方のパス区切りを許容します。

パス断片には以下のような形を使います。

```python
_PATH_SEPARATOR_PATTERN = r"[\\/]"
```

例外型だけで判定する範囲は狭くします。  
`KeyboardInterrupt` と `CancelledError` は終了・中断シグナルなので型だけで破棄できます。  
汎用的な `ValueError`、`RuntimeError`、`HTTPException`、`UnicodeDecodeError` は、型だけで破棄してはいけません。

コメントを追加する場合は、そのパターンがなぜ必要かを日本語で書きます。  
正規表現の文字列を文章で繰り返すだけのコメントは避けます。

## テスト

`test/unit/utility/test_sentry_utility.py` に代表ケースを追加します。

新しく破棄する Sentry 署名には、`test_filter_sentry_event_drops_known_unrecoverable_errors` に少なくとも1つのテストケースを追加します。

g2p / テキスト処理 / SBV2 系の例は、残す側のテストに置きます。  
残す分類から破棄分類へ移す場合は、分類を変えた理由を最終報告で説明します。

Engine 側で修正した場合は、Sentry フィルタテストだけで終わらせません。  
入力検証を強化したならモデル・API 境界のテストを追加し、例外変換を直したなら該当経路のユニットテストを追加します。

テストパスでは、`user` のような固定ユーザー名を避けます。  
インストール済みファイルには `C:\Program Files\AivisSpeech\AivisSpeech-Engine`、ユーザーデータには `C:\Users\Taro\AppData\Roaming\AivisSpeech-Engine` のような現実的な場所を使います。  
POSIX 風パスは、本番エラーが Linux やコンテナ環境から発生しうる場合だけ追加します。

## 検証

リポジトリルートから以下を実行します。

```bash
uv run ruff format --check voicevox_engine/utility/sentry_utility.py test/unit/utility/test_sentry_utility.py
uv run ruff check voicevox_engine/utility/sentry_utility.py test/unit/utility/test_sentry_utility.py
uv run pytest test/unit/utility/test_sentry_utility.py
uv run mypy voicevox_engine/utility/sentry_utility.py test/unit/utility/test_sentry_utility.py
```

フォーマット確認が落ちた場合は以下を実行します。

```bash
uv run ruff format voicevox_engine/utility/sentry_utility.py test/unit/utility/test_sentry_utility.py
```

NumPy mypy plugin の非推奨警告は、実際の型エラーとは分けて報告します。
