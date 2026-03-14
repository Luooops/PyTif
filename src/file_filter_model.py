import os
from typing import Dict, Optional, Set

from PySide6.QtCore import QModelIndex, QSortFilterProxyModel

from io_handler import SUPPORTED_EXTS
from utils import natural_key


class FileFilterModel(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._opened_files: Set[str] = set()
        self._opened_folders: Set[str] = set()
        self._excluded_paths: Set[str] = set()
        self._root_path: Optional[str] = None
        self._view_root_path: Optional[str] = None
        self._dir_match_cache: Dict[str, bool] = {}
        self.setRecursiveFilteringEnabled(True)
        self.setDynamicSortFilter(True)

    def set_sources(
        self,
        opened_files: Set[str],
        opened_folders: Set[str],
        excluded_paths: Set[str],
        root_path: Optional[str],
        view_root_path: Optional[str],
    ):
        self._opened_files = {self._norm_path(path) for path in opened_files}
        self._opened_folders = {self._norm_path(path) for path in opened_folders}
        self._excluded_paths = {self._norm_path(path) for path in excluded_paths}
        self._root_path = self._norm_path(root_path) if root_path else None
        self._view_root_path = (
            self._norm_path(view_root_path) if view_root_path else None
        )
        self._dir_match_cache.clear()
        self.invalidate()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        model = self.sourceModel()
        if model is None or self._root_path is None:
            return False

        index = model.index(source_row, 0, source_parent)
        if not index.isValid():
            return False

        path = self._norm_path(model.filePath(index))
        if path == self._view_root_path:
            return True
        if path == self._root_path:
            return True

        # Check exclusion first
        if any(
            self._is_under_or_equal(path, excluded) for excluded in self._excluded_paths
        ):
            return False

        if not self._is_under(path, self._root_path):
            return False
        if model.isDir(index):
            return self._directory_matches(path)
        return self._file_matches(path)

    def lessThan(self, source_left: QModelIndex, source_right: QModelIndex) -> bool:
        model = self.sourceModel()
        if model is None:
            return False

        left_is_dir = model.isDir(source_left)
        right_is_dir = model.isDir(source_right)
        if left_is_dir != right_is_dir:
            return left_is_dir
        return natural_key(model.fileName(source_left)) < natural_key(
            model.fileName(source_right)
        )

    def _file_matches(self, path: str) -> bool:
        if not path.lower().endswith(SUPPORTED_EXTS):
            return False
        if path in self._opened_files:
            return True
        return any(self._is_under(path, folder) for folder in self._opened_folders)

    def _directory_matches(self, path: str) -> bool:
        candidates = self._opened_files | self._opened_folders
        if any(self._is_under(candidate, path) for candidate in candidates):
            return True
        if any(self._is_under(path, folder) for folder in self._opened_folders):
            return self._directory_has_matches(path)
        return False

    def _directory_has_matches(self, path: str) -> bool:
        cached = self._dir_match_cache.get(path)
        if cached is not None:
            return cached

        try:
            with os.scandir(path) as entries:
                for entry in entries:
                    entry_path = self._norm_path(entry.path)

                    # Skip if excluded
                    if any(
                        self._is_under_or_equal(entry_path, excluded)
                        for excluded in self._excluded_paths
                    ):
                        continue

                    if entry.is_dir(follow_symlinks=False):
                        if self._directory_matches(entry_path):
                            self._dir_match_cache[path] = True
                            return True
                    elif self._file_matches(entry_path):
                        self._dir_match_cache[path] = True
                        return True
        except OSError:
            self._dir_match_cache[path] = False
            return False

        self._dir_match_cache[path] = False
        return False

    @staticmethod
    def _norm_path(path: str) -> str:
        return os.path.normcase(os.path.abspath(path))

    @staticmethod
    def _is_under(path: str, parent: str) -> bool:
        try:
            return os.path.commonpath([path, parent]) == parent and path != parent
        except ValueError:
            return False

    @staticmethod
    def _is_under_or_equal(path: str, parent: str) -> bool:
        try:
            return os.path.commonpath([path, parent]) == parent
        except ValueError:
            return False
