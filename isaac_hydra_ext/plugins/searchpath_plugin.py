"""
Hydra SearchPath плагин: добавляет pkg://isaac_hydra_ext.conf в поисковый путь,
чтобы Hydra видела YAML-файлы из этого пакета без ручных +hydra.searchpath.
"""
from hydra.plugins.search_path_plugin import SearchPathPlugin
from hydra.core.config_search_path import ConfigSearchPath

class IsaacHydraExtSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        # Добавляем корень конфигов пакета (папку conf)
        search_path.append("pkg://isaac_hydra_ext.conf")
