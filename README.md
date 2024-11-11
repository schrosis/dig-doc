# Dig Doc

Dig Doc は主に 2 つの機能を提供します。

- RAG を行うためのベクターストアの管理
- OpenAI API に互換性のある API サーバー

## ベクターストアの管理

| データソース | 実装状況 |
| --- | --- |
| ローカルファイル | ✅ |
| Confluence | ❌️ |
| Notion | ❌️ |

## API サーバー

OpenAI API に互換性のある API を提供することで OpenAI API 互換のツールに結合できるようにします。
主な用途は Cursor で、 `Override OpenAI Base URL` の設定を利用して独自の RAG を使えるようにすることです。

| 機能 | 実装状況 |
| --- | --- |
| OpenAI API 互換 API | ❌️ |
| 認証 | ❌️ |

## Docs

- [開発環境構築](docs/開発環境構築.md)
- [Taskfile](docs/taskfile.md)
- [スクリプト一覧](docs/scripts/index.md)
