#!/bin/bash
# Заливка недостающих датасетов на аккаунты Kaggle, все параллельно.
# Запуск: ./upload_all.sh   (VPN должен быть ВЫКЛЮЧЕН — иначе вместо минут будут часы)
cd /Users/kitty/PivoOzon || exit 1
K=".venv/bin/kaggle"
mkdir -p output/uploads

SEMEN="KGAT_be2188ed27d348d20eafed09583d1f3b"
DARK="KGAT_280ad1bc587765ad26c193f29cfc2182"
YLL="KGAT_4817db494f69d2c36d9c876c12dde45b"

up() {  # up <токен> <каталог> <имя-лога>
  KAGGLE_API_TOKEN="$1" KAGGLE_CONFIG_DIR="$HOME/.kaggle_semen" \
    $K datasets create -p "$2" > "output/uploads/$3.log" 2>&1
  if grep -q "is being created" "output/uploads/$3.log"; then
    echo "  OK   $3"
  else
    echo "  СБОЙ $3 — см. output/uploads/$3.log"
  fi
}

echo "Заливка пошла, всё параллельно..."
up "$SEMEN" output/kaggle/semenmaskviten_stage1      semen_stage1 &
up "$DARK"  output/kaggle/darkdustrydarkness_data    dark_data    &
up "$DARK"  output/kaggle/darkdustrydarkness_src     dark_src     &
up "$YLL"   output/kaggle/yllxio_data                yll_data     &
up "$YLL"   output/kaggle/yllxio_src                 yll_src      &
up "$YLL"   output/kaggle/yllxio_stage1              yll_stage1   &
wait
echo "Готово. Логи: output/uploads/"
