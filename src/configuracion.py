import yaml


def cargar_configuracion(ruta_config='config.yaml'):
    with open(ruta_config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config
