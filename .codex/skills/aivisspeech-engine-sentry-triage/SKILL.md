---
name: aivisspeech-engine-sentry-triage
description: AivisSpeech Engine の Sentry issue を調査し、修正すべきエンジン側の不具合と、入力値・ローカル環境・外部サービス由来のノイズを切り分けるためのスキルです。Sentry 側で既知ノイズを永続アーカイブする作業や、voicevox_engine/utility/sentry_utility.py と関連テストを更新して既知ノイズを送信前に破棄する作業で使用します。
---

# AivisSpeech Engine Sentry トリアージ

## 作業手順

このスキルは、AivisSpeech Engine で今後も繰り返す Sentry ノイズ処理の流れで使用します。

1. Sentry の実データから issue とイベントを確認します
2. 各 issue を「修正対象」「入力値ノイズ」「ローカル環境ノイズ」「外部サービスノイズ」「保留」に分類します
3. ユーザーが Sentry 側の整理を依頼した場合だけ、確定ノイズの issue を永続アーカイブします
4. 同じノイズを送信前に破棄するため、`voicevox_engine/utility/sentry_utility.py` を更新します
5. `test/unit/utility/test_sentry_utility.py` に絞り込みテストを追加します
6. 修正すべきエラーは、修正先がこのリポジトリ内かどうかを切り分けます
7. リポジトリルートから `uv run` の検証コマンドを実行します

フィルタを変更する前に、必ず Sentry の実 issue / イベントを確認します。  
イベント詳細が取得できる場合は、タイトルだけで分類しないでください。

## 修正判断

AivisSpeech Engine 側で完結する問題は、ユーザーの追加承認を待たずに修正します。  
たとえば API バリデーション不足、`AudioQuery` モデルの検証不足、Engine 側の例外処理・レスポンス変換・Sentry フィルタ漏れ、Engine 側で管理している辞書データの不整合は、このリポジトリで実装とテストまで進めます。

SBV2 コア、依存ライブラリ、配布モデル、別リポジトリの辞書データに修正が必要な場合は、このリポジトリだけで直せるように見せかけません。  
Sentry issue、最新イベント、該当スタック、再現条件、修正候補、修正先リポジトリを短くまとめて、ユーザーへ対応依頼として報告します。

修正先が曖昧な場合は、まずこのリポジトリ内で入力検証・前処理・呼び出し境界を確認します。  
Engine 側で不正入力を止められるなら Engine 側で直し、コア内部の正常入力処理で壊れている場合はユーザーへ詳細報告します。

## 必読資料

作業前に以下を読みます。

- `references/sentry-workflow.md`: Sentry API の使い分け、90日 issue 一覧、アーカイブ時の注意
- `references/classification.md`: 既存調査に基づく「破棄」「残す」「保留」の分類ルール
- `references/filter-implementation.md`: このリポジトリでの実装・テスト方針

タイトルベースの粗い確認だけでよい場面では、`scripts/probe_sentry_filter.py` を使えます。

```bash
uv run python .codex/skills/aivisspeech-engine-sentry-triage/scripts/probe_sentry_filter.py \
  --repo-root . \
  --query "is:ignored" \
  --stats-period 90d
```

このスクリプトは読み取り専用です。  
最終的な「残す / 破棄する」判断の前にはイベント詳細を確認してください。

## リポジトリ前提

コマンドは AivisSpeech Engine のリポジトリルートから実行します。

Python 実行には `uv run python` を使います。  
グローバルの `python` や `python3` は絶対に使いません。

バンドル済み Sentry helper で取得できる内容は、まず helper を使います。  
helper で表現できないクエリが必要な場合は、`curl` または `uv run python` で Sentry REST API を直接叩きます。  
ただし、ユーザーが Sentry issue 状態の更新を明示した場合を除き、読み取り専用のリクエストに限定します。

`SENTRY_AUTH_TOKEN` は絶対に出力しません。

ユーザーが明示しない限り、ステージング、ステージング解除、コミット、リセット、クリーンアップ、ファイル削除は行いません。
