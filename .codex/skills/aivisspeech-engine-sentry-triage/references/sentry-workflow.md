# Sentry 作業手順

## API の使い分け

まずバンドル済みの Sentry helper を使います。

```bash
uv run python /Users/tsukumi/.codex/plugins/cache/openai-curated-remote/sentry/0.1.2/skills/sentry/scripts/sentry_api.py \
  --org aivis-project \
  --project aivisspeech-engine \
  list-issues \
  --query "is:unresolved" \
  --limit 100 \
  --time-range 14d
```

helper の `list-issues` コマンドはプロジェクト別 issue API を使います。  
この API の `statsPeriod` は `""`、`24h`、`14d` だけを受け付けます。

Sentry の UI に近い90日 issue 一覧が必要な場合は、組織別 issue API を使います。

```bash
curl -sS \
  -H "Authorization: Bearer $SENTRY_AUTH_TOKEN" \
  "https://sentry.io/api/0/organizations/aivis-project/issues/?project=4508555159470080&query=is%3Aunresolved&limit=100&statsPeriod=90d"
```

トークンは出力しません。  
環境変数の確認が必要な場合も、トークンが設定されているかどうかだけを表示します。

## イベント詳細の確認

Sentry の issue タイトルは、省略されたり正規化されたりします。  
フィルタ漏れを判断する前に、少なくとも最新イベントを取得します。

1. 組織別 API から issue 一覧を取得します
2. `/api/0/organizations/aivis-project/issues/{issue_id}/events/` から最新イベントを取得します
3. `/api/0/projects/aivis-project/aivisspeech-engine/events/{event_id}/` からイベント詳細を取得します
4. `metadata`、`entries[type=exception]`、`culprit`、リクエストパスを確認します

イベント詳細からメモリ・GPU・ローカルファイル・ダウンロード・終了処理ノイズだと確認できた場合は、フィルタとテストを追加します。  
g2p、テキスト前処理、SBV2、ONNX 入力形状など、エンジン側の問題である可能性が残る場合は、ユーザーが方針変更を明示しない限り残します。

## Issue のアーカイブ

ユーザーが Sentry issue 状態の更新を依頼した場合だけ、issue をアーカイブします。

削除は使わず、`ignored / archived forever` を使います。  
削除すると、同じグループが後から新規 issue として再登場する可能性があります。

確定ノイズに使う更新本文は以下です。

```json
{"status":"ignored","substatus":"archived_forever","statusDetails":{}}
```

まず1件だけで書き込みテストを行い、その後で確定グループだけを一括処理します。

## プロジェクト値

- 組織: `aivis-project`
- プロジェクト slug: `aivisspeech-engine`
- 調査時に確認したプロジェクト ID: `4508555159470080`
- Sentry フィルタ実装: `voicevox_engine/utility/sentry_utility.py`
- Sentry フィルタテスト: `test/unit/utility/test_sentry_utility.py`
