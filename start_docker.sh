#!/bin/bash

# Скрипт для сборки Docker-образа и запуска интерактивной сессии внутри контейнера.
# Рабочая директория внутри контейнера будет /code,
# которая связана с вашей локальной папкой aidao.

echo "--- Шаг 1: Установка nvidia-container-toolkit для использования gpu в docker контейнере ---"
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

echo "--- Шаг 2: Сборка Docker-образа 'aidao-container' ---"
docker compose build

# Проверка, успешно ли собрался образ
if [ $? -ne 0 ]; then
    echo "Ошибка: не удалось собрать Docker-образ. Пожалуйста, проверьте Dockerfile и docker-compose.yml."
    exit 1
fi

echo ""
echo "--- Шаг 2: Запуск интерактивной сессии в контейнере ---"
echo "Вы будете перемещены в командную строку внутри контейнера."
echo "Рабочая директория: /code"
echo ""
echo "Доступные команды для запуска скриптов:"


docker compose run --rm aidao /bin/bash

echo "--- Сессия в контейнере завершена ---"
