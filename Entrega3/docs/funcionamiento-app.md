# Guía de Instalación y Ejecución de la Aplicación Flask

Este documento describe los pasos necesarios para configurar el entorno de desarrollo y ejecutar la aplicación tanto en macOS como en windows.


## Paso 1: Instalar Herramientas Base (Homebrew y Python 3.10)

Si ya tienes Homebrew y Python 3.10 instalados, puedes saltar a la siguiente sección.

### 1.1. Instalar Homebrew

Homebrew es un gestor de paquetes que facilita la instalación de software en macOS.

-   Abre la Terminal y ejecuta el siguiente comando para instalarlo:
    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```

### 1.2. Instalar Python 3.10

Usaremos Homebrew para instalar la versión específica de Python requerida por el proyecto.

-   En la Terminal, ejecuta:
    ```bash
    brew install python@3.10
    ```

-   Verifica que la instalación fue exitosa:
    ```bash
    python3.10 --version
    ```
    Deberías ver una salida como `Python 3.10.x`.

en windows usa:

```bash
winget install Python.Python.3.10
```
---


## Paso 2: Configurar el Entorno Virtual

Un entorno virtual aísla las dependencias del proyecto para evitar conflictos con otros proyectos.

### 2.1. Crear el Entorno Virtual

-   Usa el módulo `venv` de Python 3.10 para crear un entorno llamado `venv`:
    ```bash
    python3.10 -m venv venv
    ```
en windows:
```bash
python -m venv venv
```

### 2.2. Activar el Entorno Virtual

-   Para empezar a usar el entorno, debes activarlo:
    ```bash
    source venv/bin/activate
    ```
-   Sabrás que está activo porque el prompt de tu terminal cambiará para mostrar `(venv)` al principio.

---
en windows:
```bash
.\venv\Scripts\activate
```
## Paso 3: Instalar las Dependencias

### 3.1. Instalar desde `requirements.txt`

-   El archivo `requirements.txt` contiene la lista de todas las librerías que el proyecto necesita. Instálalas con `pip`:
    ```bash
    pip install -r requirements.txt
    ```
en windows es igual

### 3.2. Solución a Warning de Versiones (scikit-learn)

-   Es posible que al instalar las dependencias, la versión de `scikit-learn` no sea la misma con la que se guardaron los modelos de Machine Learning, lo que causa un `InconsistentVersionWarning`.
-   Para asegurar la compatibilidad y evitar resultados incorrectos, instala la versión exacta con la que se crearon los modelos:
    ```bash
    pip install scikit-learn==1.6.1
    ```
    *Nota: Si en el futuro los modelos se re-entrenan con una versión más nueva, este comando deberá actualizarse.*

en windows es igual

---

## Paso 5: Ejecutar la Aplicación Flask

Con el entorno activado y las dependencias instaladas, ya puedes iniciar el servidor de desarrollo.

### 5.1. Iniciar el Servidor

-   Para correr la aplicación en **modo de depuración** (recomendado para desarrollo, ya que se reinicia automáticamente al detectar cambios en el código), usa el siguiente comando:
    ```bash
    python app.py
    ```

### 5.2. Ver la Aplicación

-   La terminal te mostrará un mensaje indicando que el servidor está corriendo, similar a este:
    ```
     * Running on http://127.0.0.1:5000
    Press CTRL+C to quit
    ```
-   Abre tu navegador web y ve a la dirección **http://127.0.0.1:5000**.

### 5.3. Detener el Servidor

-   Para detener la aplicación, regresa a la ventana de la terminal y presiona las teclas **`Ctrl + C`**.

---

