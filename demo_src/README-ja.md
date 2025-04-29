# FastAPI Admin

[![image](https://img.shields.io/pypi/v/fastapi-admin.svg?style=flat)](https://pypi.python.org/pypi/fastapi-admin)
[![image](https://img.shields.io/github/license/fastapi-admin/fastapi-admin)](https://github.com/fastapi-admin/fastapi-admin)
[![image](https://github.com/fastapi-admin/fastapi-admin/workflows/deploy/badge.svg)](https://github.com/fastapi-admin/fastapi-admin/actions?query=workflow:deploy)
[![image](https://github.com/fastapi-admin/fastapi-admin/workflows/pypi/badge.svg)](https://github.com/fastapi-admin/fastapi-admin/actions?query=workflow:pypi)

[中文文档](./README-zh.md)
[한국어 문서](./README-ko.md)
[日本語ドキュメント](./README-ja.md)

## はじめに

`fastapi-admin`は、[tabler](https://github.com/tabler/tabler)UIを使用して開発された[FastAPI](https://github.com/tiangolo/fastapi)と[TortoiseORM](https://github.com/tortoise/tortoise-orm/)の高速な管理ダッシュボードで、Django adminを参考にしています。

## インストール

```shell
> pip install fastapi-admin
```

## 環境

- [Redis](https://redis.io)

## オンラインデモ

[ここ](https://fastapi-admin.long2ice.io/admin/login)でオンラインデモを確認できます。

- username: `admin`
- password: `123456`

もしくは、[ここ](https://fastapi-admin-pro.long2ice.io/admin/login)でプロ版のオンラインデモを確認できます。

- username: `admin`
- password: `123456`

## スクリーンショット

![](https://raw.githubusercontent.com/fastapi-admin/fastapi-admin/dev/images/login.png)

![](https://raw.githubusercontent.com/fastapi-admin/fastapi-admin/dev/images/dashboard.png)

## ローカル環境での実行方法

1. レポジトリをクローンする。
2. .envファイルを作成し、以下の内容を記述する。

   ```dotenv
   DATABASE_URL=mysql://root:123456@127.0.0.1:3306/fastapi-admin
   REDIS_URL=redis://localhost:6379/0
   ```

3. `docker-compose up -d --build`を実行する。
4. <http://localhost:8000/admin/init>にアクセスして、管理者用アカウントを作成する。

## ドキュメント

ドキュメントは[こちら](https://fastapi-admin-docs.long2ice.io)で確認できます。

## ライセンス

このプロジェクトは、[Apache-2.0](https://github.com/fastapi-admin/fastapi-admin/blob/master/LICENSE)ライセンスに準拠しています。
