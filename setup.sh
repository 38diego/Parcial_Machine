#!/bin/bash
echo "Actualizando lista de paquetes..."
apt-get update -y

echo "Instalando R..."
apt-get install -y r-base

echo "Verificando instalación de Rscript..."
if [ -f /usr/bin/Rscript ]; then
    echo "Rscript encontrado en /usr/bin/Rscript"
else
    echo "Rscript no se encontró en /usr/bin/"
    exit 1
fi
