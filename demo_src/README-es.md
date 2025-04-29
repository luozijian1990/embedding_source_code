# FastAPI Admin

[![image](https://img.shields.io/pypi/v/fastapi-admin.svg?style=flat)](https://pypi.python.org/pypi/fastapi-admin)
[![image](https://img.shields.io/github/license/fastapi-admin/fastapi-admin)](https://github.com/fastapi-admin/fastapi-admin)
[![image](https://github.com/fastapi-admin/fastapi-admin/workflows/deploy/badge.svg)](https://github.com/fastapi-admin/fastapi-admin/actions?query=workflow:deploy)
[![image](https://github.com/fastapi-admin/fastapi-admin/workflows/pypi/badge.svg)](https://github.com/fastapi-admin/fastapi-admin/actions?query=workflow:pypi)

[中文文档](./README-zh.md)
[한국어 문서](./README-ko.md)

## Introducción

`fastapi-admin` es un panel de administrador basado en [FastAPI](https://github.com/tiangolo/fastapi) y [TortoiseORM](https://github.com/tortoise/tortoise-orm/) con [tabler](https://github.com/tabler/tabler) ui, inspirado en Djando Admin.

## Instalación

```shell
> pip install fastapi-admin
```

## Requisitos

- [Redis](https://redis.io)

## Demo Online

Puedes checar una demo online [aquí](https://fastapi-admin.long2ice.io/admin/login).

- Usuario: `admin`
- Contraseña: `123456`

O una demo online versión pro [aquí](https://fastapi-admin-pro.long2ice.io/admin/login).

- Usuario: `admin`
- Contraseña: `123456`

## Capturas

![](https://raw.githubusercontent.com/fastapi-admin/fastapi-admin/dev/images/login.png)

![](https://raw.githubusercontent.com/fastapi-admin/fastapi-admin/dev/images/dashboard.png)

## Ejecutar el ejemplo localmente

1. Clona el repositorio.
2. Crea un archivo `.env`.
   ```dotenv
   DATABASE_URL=mysql://root:123456@127.0.0.1:3306/fastapi_admin
   REDIS_URL=redis://localhost:6379/0
   ```
3. Ejecuta el comando `docker-compose up -d --build`.
4. Visita <http://localhost:8000/admin/init> para crear el primer administrador.

## Documentación

Mira la documentación en <https://fastapi-admin-docs.long2ice.io>.

## Licencia

Este proyecto está licenciado bajo la [licencia Apache-2.0](https://github.com/fastapi-admin/fastapi-admin/blob/master/LICENSE).
