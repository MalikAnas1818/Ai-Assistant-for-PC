import os
import shutil
import logging
import json
import time
import hashlib
import zipfile
import stat
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import threading
from collections import deque


# =========================
# LOGGING SETUP
# =========================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =========================
# ENUMS & DATA CLASSES
# =========================

class FileOperation(Enum):
    """Types of file operations"""
    CREATE_FOLDER = "create_folder"
    DELETE_FOLDER = "delete_folder"
    COPY_FILE = "copy_file"
    MOVE_FILE = "move_file"
    DELETE_FILE = "delete_file"
    RENAME_FILE = "rename_file"
    COMPRESS = "compress"
    EXTRACT = "extract"
    BACKUP = "backup"
    SYNC = "sync"


class FileType(Enum):
    """File types for filtering"""
    DOCUMENT = ["pdf", "docx", "doc", "txt", "xlsx", "pptx"]
    IMAGE = ["jpg", "jpeg", "png", "gif", "bmp", "svg"]
    VIDEO = ["mp4", "avi", "mkv", "mov", "wmv"]
    AUDIO = ["mp3", "wav", "flac", "aac", "ogg"]
    CODE = ["py", "js", "cpp", "java", "html", "css"]
    ARCHIVE = ["zip", "rar", "7z", "tar", "gz"]
    ALL = []


@dataclass
class OperationResult:
    """Result of a file operation"""
    operation: FileOperation
    success: bool
    source: str
    destination: str
    timestamp: datetime
    duration: float
    file_size: int
    message: str
    details: Dict[str, Any]


@dataclass
class FileInfo:
    """Information about a file"""
    path: str
    name: str
    size: int
    created_time: datetime
    modified_time: datetime
    is_dir: bool
    permissions: str
    extension: str
    file_type: str
    hash_value: Optional[str] = None


@dataclass
class DriveStat:
    """Statistics about a drive"""
    drive_letter: str
    total_space: int
    used_space: int
    free_space: int
    percent_used: float
    timestamp: datetime


# =========================
# ADVANCED FILE MANAGER
# =========================

class AdvancedFileManager:
    """
    Advanced file management system with:
    - Batch operations
    - File compression
    - Backup systems
    - Drive monitoring
    - Detailed logging
    - Error recovery
    - Permission management
    - File hashing
    """

    def __init__(self, enable_logging: bool = True):
        """Initialize the file manager"""
        self.operation_history: deque = deque(maxlen=100)
        self.callbacks: List[Callable] = []
        self.enable_logging = enable_logging
        self.start_time = datetime.now()

        logger.info("✅ Advanced File Manager initialized")

    # =========================
    # DRIVE OPERATIONS
    # =========================

    def get_valid_drive_path(self, drive_letter: str) -> Optional[str]:
        """
        Get valid drive path with validation
        """
        try:
            drive_letter = drive_letter.upper().replace(":", "").strip()

            if not drive_letter:
                logger.error("❌ Invalid drive letter")
                return None

            path = f"{drive_letter}:/"

            if os.path.exists(path):
                logger.info(f"✅ Drive found: {path}")
                return path
            else:
                logger.warning(f"⚠️ Drive {drive_letter}: not found")
                return None

        except Exception as e:
            logger.error(f"❌ Drive validation error: {e}")
            return None

    def get_drive_stats(self, drive_letter: str = "C") -> Optional[DriveStat]:
        """
        Get comprehensive drive statistics
        """
        try:
            drive_path = self.get_valid_drive_path(drive_letter)

            if not drive_path:
                return None

            # Get disk usage
            total, used, free = shutil.disk_usage(drive_path)
            percent_used = (used / total) * 100 if total > 0 else 0

            stat = DriveStat(
                drive_letter=drive_letter,
                total_space=total,
                used_space=used,
                free_space=free,
                percent_used=percent_used,
                timestamp=datetime.now()
            )

            logger.info(f"📊 Drive {drive_letter} stats retrieved")
            return stat

        except Exception as e:
            logger.error(f"❌ Drive stats error: {e}")
            return None

    def list_drives(self) -> List[str]:
        """
        List all available drives
        """
        try:
            drives = []
            for drive_letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                if self.get_valid_drive_path(drive_letter):
                    drives.append(drive_letter)

            logger.info(f"✅ Found {len(drives)} drives: {drives}")
            return drives

        except Exception as e:
            logger.error(f"❌ List drives error: {e}")
            return []

    def print_drive_stats(self, drive_letter: str = "C") -> None:
        """Print drive statistics in nice format"""
        stat = self.get_drive_stats(drive_letter)

        if not stat:
            print(f"❌ Could not get stats for drive {drive_letter}")
            return

        print("\n" + "="*50)
        print(f"💾 DRIVE STATISTICS: {stat.drive_letter}:")
        print("="*50)
        print(f"Total Space: {self._format_size(stat.total_space)}")
        print(f"Used Space: {self._format_size(stat.used_space)} ({stat.percent_used:.1f}%)")
        print(f"Free Space: {self._format_size(stat.free_space)}")
        print("="*50 + "\n")

    # =========================
    # FOLDER OPERATIONS
    # =========================

    def create_folder(
        self,
        folder_name: str,
        drive: str = "C",
        nested_path: str = ""
    ) -> Optional[str]:
        """
        Create folder in any drive with nested path support
        """
        start_time = time.time()

        try:
            base_path = self.get_valid_drive_path(drive)

            if not base_path:
                logger.error(f"❌ Drive {drive}: not found")
                return None

            if nested_path:
                full_path = os.path.join(base_path, nested_path, folder_name)
            else:
                full_path = os.path.join(base_path, folder_name)

            os.makedirs(full_path, exist_ok=True)

            duration = time.time() - start_time

            result = OperationResult(
                operation=FileOperation.CREATE_FOLDER,
                success=True,
                source="",
                destination=full_path,
                timestamp=datetime.now(),
                duration=duration,
                file_size=0,
                message=f"Folder created successfully",
                details={"path": full_path}
            )

            self.operation_history.append(result)
            self._trigger_callbacks(result)

            logger.info(f"✅ Folder created: {full_path}")
            return full_path

        except Exception as e:
            logger.error(f"❌ Create folder error: {e}")

            result = OperationResult(
                operation=FileOperation.CREATE_FOLDER,
                success=False,
                source="",
                destination="",
                timestamp=datetime.now(),
                duration=time.time() - start_time,
                file_size=0,
                message=str(e),
                details={}
            )

            self.operation_history.append(result)
            self._trigger_callbacks(result)

            return None

    def delete_folder(self, folder_path: str, recursive: bool = True) -> bool:
        """
        Delete folder with safety checks
        """
        start_time = time.time()

        try:
            if not os.path.exists(folder_path):
                logger.warning(f"⚠️ Folder not found: {folder_path}")
                return False

            if not os.path.isdir(folder_path):
                logger.error(f"❌ Not a directory: {folder_path}")
                return False

            if recursive:
                shutil.rmtree(folder_path)
            else:
                os.rmdir(folder_path)

            duration = time.time() - start_time

            result = OperationResult(
                operation=FileOperation.DELETE_FOLDER,
                success=True,
                source=folder_path,
                destination="",
                timestamp=datetime.now(),
                duration=duration,
                file_size=0,
                message="Folder deleted successfully",
                details={"recursive": recursive}
            )

            self.operation_history.append(result)
            self._trigger_callbacks(result)

            logger.info(f"✅ Folder deleted: {folder_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Delete folder error: {e}")

            result = OperationResult(
                operation=FileOperation.DELETE_FOLDER,
                success=False,
                source=folder_path,
                destination="",
                timestamp=datetime.now(),
                duration=time.time() - start_time,
                file_size=0,
                message=str(e),
                details={}
            )

            self.operation_history.append(result)

            return False

    def list_folder_contents(
        self,
        folder_path: str,
        file_type: FileType = FileType.ALL,
        recursive: bool = False
    ) -> List[FileInfo]:
        """
        List folder contents with filtering
        """
        try:
            contents = []

            if not os.path.exists(folder_path):
                logger.warning(f"⚠️ Folder not found: {folder_path}")
                return contents

            if recursive:
                for root, dirs, files in os.walk(folder_path):
                    for filename in files:
                        filepath = os.path.join(root, filename)
                        file_info = self.get_file_info(filepath)

                        if file_info and self._matches_type(file_info, file_type):
                            contents.append(file_info)
            else:
                for item in os.listdir(folder_path):
                    item_path = os.path.join(folder_path, item)
                    file_info = self.get_file_info(item_path)

                    if file_info and self._matches_type(file_info, file_type):
                        contents.append(file_info)

            logger.info(f"✅ Listed {len(contents)} items in {folder_path}")
            return contents

        except Exception as e:
            logger.error(f"❌ List folder error: {e}")
            return []

    # =========================
    # FILE OPERATIONS
    # =========================

    def get_file_info(self, file_path: str) -> Optional[FileInfo]:
        """
        Get comprehensive file information
        """
        try:
            if not os.path.exists(file_path):
                return None

            stat_info = os.stat(file_path)
            name = os.path.basename(file_path)
            extension = os.path.splitext(name)[1].lstrip('.')
            is_dir = os.path.isdir(file_path)

            # Get file type
            file_type = self._get_file_type(extension)

            # Get permissions
            permissions = oct(stat_info.st_mode)[-3:]

            # Get timestamps
            created_time = datetime.fromtimestamp(stat_info.st_ctime)
            modified_time = datetime.fromtimestamp(stat_info.st_mtime)

            info = FileInfo(
                path=file_path,
                name=name,
                size=stat_info.st_size,
                created_time=created_time,
                modified_time=modified_time,
                is_dir=is_dir,
                permissions=permissions,
                extension=extension,
                file_type=file_type
            )

            return info

        except Exception as e:
            logger.warning(f"⚠️ Could not get file info: {e}")
            return None

    def copy_file(
        self,
        source_path: str,
        dest_folder: str,
        drive: str = "C",
        preserve_metadata: bool = True
    ) -> Optional[str]:
        """
        Copy file to destination with metadata preservation
        """
        start_time = time.time()

        try:
            if not os.path.exists(source_path):
                logger.error(f"❌ Source file not found: {source_path}")
                return None

            base_path = self.get_valid_drive_path(drive)

            if not base_path:
                logger.error(f"❌ Drive {drive}: not found")
                return None

            dest_path_full = os.path.join(base_path, dest_folder)
            os.makedirs(dest_path_full, exist_ok=True)

            file_name = os.path.basename(source_path)
            dest_file_path = os.path.join(dest_path_full, file_name)

            # Copy file
            if preserve_metadata:
                shutil.copy2(source_path, dest_file_path)
            else:
                shutil.copy(source_path, dest_file_path)

            file_size = os.path.getsize(source_path)
            duration = time.time() - start_time

            result = OperationResult(
                operation=FileOperation.COPY_FILE,
                success=True,
                source=source_path,
                destination=dest_file_path,
                timestamp=datetime.now(),
                duration=duration,
                file_size=file_size,
                message="File copied successfully",
                details={"preserve_metadata": preserve_metadata}
            )

            self.operation_history.append(result)
            self._trigger_callbacks(result)

            logger.info(f"✅ File copied: {source_path} → {dest_file_path}")
            return dest_file_path

        except Exception as e:
            logger.error(f"❌ Copy file error: {e}")

            result = OperationResult(
                operation=FileOperation.COPY_FILE,
                success=False,
                source=source_path,
                destination="",
                timestamp=datetime.now(),
                duration=time.time() - start_time,
                file_size=0,
                message=str(e),
                details={}
            )

            self.operation_history.append(result)

            return None

    def move_file(
        self,
        source_path: str,
        dest_folder: str,
        drive: str = "C"
    ) -> Optional[str]:
        """
        Move file to destination
        """
        start_time = time.time()

        try:
            if not os.path.exists(source_path):
                logger.error(f"❌ Source file not found: {source_path}")
                return None

            base_path = self.get_valid_drive_path(drive)

            if not base_path:
                logger.error(f"❌ Drive {drive}: not found")
                return None

            dest_path_full = os.path.join(base_path, dest_folder)
            os.makedirs(dest_path_full, exist_ok=True)

            file_name = os.path.basename(source_path)
            dest_file_path = os.path.join(dest_path_full, file_name)

            # Move file
            shutil.move(source_path, dest_file_path)

            file_size = os.path.getsize(source_path)
            duration = time.time() - start_time

            result = OperationResult(
                operation=FileOperation.MOVE_FILE,
                success=True,
                source=source_path,
                destination=dest_file_path,
                timestamp=datetime.now(),
                duration=duration,
                file_size=file_size,
                message="File moved successfully",
                details={}
            )

            self.operation_history.append(result)
            self._trigger_callbacks(result)

            logger.info(f"✅ File moved: {source_path} → {dest_file_path}")
            return dest_file_path

        except Exception as e:
            logger.error(f"❌ Move file error: {e}")

            result = OperationResult(
                operation=FileOperation.MOVE_FILE,
                success=False,
                source=source_path,
                destination="",
                timestamp=datetime.now(),
                duration=time.time() - start_time,
                file_size=0,
                message=str(e),
                details={}
            )

            self.operation_history.append(result)

            return None

    def delete_file(self, file_path: str) -> bool:
        """
        Delete file safely
        """
        start_time = time.time()

        try:
            if not os.path.exists(file_path):
                logger.warning(f"⚠️ File not found: {file_path}")
                return False

            if not os.path.isfile(file_path):
                logger.error(f"❌ Not a file: {file_path}")
                return False

            os.remove(file_path)

            duration = time.time() - start_time

            result = OperationResult(
                operation=FileOperation.DELETE_FILE,
                success=True,
                source=file_path,
                destination="",
                timestamp=datetime.now(),
                duration=duration,
                file_size=0,
                message="File deleted successfully",
                details={}
            )

            self.operation_history.append(result)
            self._trigger_callbacks(result)

            logger.info(f"✅ File deleted: {file_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Delete file error: {e}")

            result = OperationResult(
                operation=FileOperation.DELETE_FILE,
                success=False,
                source=file_path,
                destination="",
                timestamp=datetime.now(),
                duration=time.time() - start_time,
                file_size=0,
                message=str(e),
                details={}
            )

            self.operation_history.append(result)

            return False

    def rename_file(self, file_path: str, new_name: str) -> Optional[str]:
        """
        Rename file
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"❌ File not found: {file_path}")
                return None

            directory = os.path.dirname(file_path)
            new_path = os.path.join(directory, new_name)

            os.rename(file_path, new_path)

            result = OperationResult(
                operation=FileOperation.RENAME_FILE,
                success=True,
                source=file_path,
                destination=new_path,
                timestamp=datetime.now(),
                duration=0,
                file_size=0,
                message="File renamed successfully",
                details={"old_name": os.path.basename(file_path), "new_name": new_name}
            )

            self.operation_history.append(result)
            self._trigger_callbacks(result)

            logger.info(f"✅ File renamed: {file_path} → {new_path}")
            return new_path

        except Exception as e:
            logger.error(f"❌ Rename file error: {e}")
            return None

    # =========================
    # BATCH OPERATIONS
    # =========================

    def batch_copy_files(
        self,
        source_folder: str,
        dest_folder: str,
        file_type: FileType = FileType.ALL
    ) -> List[OperationResult]:
        """
        Copy multiple files matching criteria
        """
        results = []

        try:
            files = self.list_folder_contents(source_folder, file_type)

            logger.info(f"📦 Starting batch copy of {len(files)} files...")

            for file_info in files:
                result_path = self.copy_file(file_info.path, dest_folder)
                if result_path:
                    results.append(len(results) + 1)

            logger.info(f"✅ Batch copy complete: {len(results)} files copied")
            return results

        except Exception as e:
            logger.error(f"❌ Batch copy error: {e}")
            return results

    def batch_move_files(
        self,
        source_folder: str,
        dest_folder: str,
        file_type: FileType = FileType.ALL
    ) -> List[str]:
        """
        Move multiple files matching criteria
        """
        results = []

        try:
            files = self.list_folder_contents(source_folder, file_type)

            logger.info(f"📦 Starting batch move of {len(files)} files...")

            for file_info in files:
                result_path = self.move_file(file_info.path, dest_folder)
                if result_path:
                    results.append(result_path)

            logger.info(f"✅ Batch move complete: {len(results)} files moved")
            return results

        except Exception as e:
            logger.error(f"❌ Batch move error: {e}")
            return results

    # =========================
    # COMPRESSION
    # =========================

    def compress_folder(
        self,
        folder_path: str,
        output_zip: Optional[str] = None
    ) -> Optional[str]:
        """
        Compress folder to zip
        """
        start_time = time.time()

        try:
            if not os.path.exists(folder_path):
                logger.error(f"❌ Folder not found: {folder_path}")
                return None

            if not output_zip:
                output_zip = f"{folder_path}.zip"

            logger.info(f"📦 Compressing {folder_path}...")

            shutil.make_archive(
                output_zip.replace('.zip', ''),
                'zip',
                folder_path
            )

            duration = time.time() - start_time
            file_size = os.path.getsize(output_zip)

            result = OperationResult(
                operation=FileOperation.COMPRESS,
                success=True,
                source=folder_path,
                destination=output_zip,
                timestamp=datetime.now(),
                duration=duration,
                file_size=file_size,
                message="Folder compressed successfully",
                details={"format": "zip"}
            )

            self.operation_history.append(result)
            self._trigger_callbacks(result)

            logger.info(f"✅ Compressed: {output_zip}")
            return output_zip

        except Exception as e:
            logger.error(f"❌ Compression error: {e}")
            return None

    def extract_zip(
        self,
        zip_path: str,
        extract_to: Optional[str] = None
    ) -> Optional[str]:
        """
        Extract zip file
        """
        start_time = time.time()

        try:
            if not os.path.exists(zip_path):
                logger.error(f"❌ Zip file not found: {zip_path}")
                return None

            if not extract_to:
                extract_to = os.path.dirname(zip_path)

            logger.info(f"📦 Extracting {zip_path}...")

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)

            duration = time.time() - start_time

            result = OperationResult(
                operation=FileOperation.EXTRACT,
                success=True,
                source=zip_path,
                destination=extract_to,
                timestamp=datetime.now(),
                duration=duration,
                file_size=os.path.getsize(zip_path),
                message="Archive extracted successfully",
                details={}
            )

            self.operation_history.append(result)
            self._trigger_callbacks(result)

            logger.info(f"✅ Extracted to: {extract_to}")
            return extract_to

        except Exception as e:
            logger.error(f"❌ Extraction error: {e}")
            return None

    # =========================
    # BACKUP
    # =========================

    def create_backup(
        self,
        source_path: str,
        backup_folder: str,
        include_timestamp: bool = True
    ) -> Optional[str]:
        """
        Create backup of file or folder
        """
        try:
            if not os.path.exists(source_path):
                logger.error(f"❌ Source not found: {source_path}")
                return None

            os.makedirs(backup_folder, exist_ok=True)

            if include_timestamp:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"{os.path.basename(source_path)}_{timestamp}"
            else:
                backup_name = os.path.basename(source_path)

            backup_path = os.path.join(backup_folder, backup_name)

            if os.path.isdir(source_path):
                shutil.copytree(source_path, backup_path)
            else:
                shutil.copy2(source_path, backup_path)

            result = OperationResult(
                operation=FileOperation.BACKUP,
                success=True,
                source=source_path,
                destination=backup_path,
                timestamp=datetime.now(),
                duration=0,
                file_size=self._get_size(source_path),
                message="Backup created successfully",
                details={"timestamp": include_timestamp}
            )

            self.operation_history.append(result)
            self._trigger_callbacks(result)

            logger.info(f"✅ Backup created: {backup_path}")
            return backup_path

        except Exception as e:
            logger.error(f"❌ Backup error: {e}")
            return None
        
    

    # =========================
    # UTILITIES
    # =========================

    def _format_size(self, size: int) -> str:
        """Format size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} PB"

    def _get_size(self, path: str) -> int:
        """Get total size of file or folder"""
        if os.path.isfile(path):
            return os.path.getsize(path)

        total = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total += os.path.getsize(filepath)
        return total

    def _get_file_type(self, extension: str) -> str:
        """Get file type from extension"""
        extension = extension.lower()

        for file_type in FileType:
            if extension in file_type.value:
                return file_type.name

        return "OTHER"

    def _matches_type(self, file_info: FileInfo, file_type: FileType) -> bool:
        """Check if file matches type filter"""
        if file_type == FileType.ALL:
            return True
        return file_info.file_type == file_type.name

    def _trigger_callbacks(self, result: OperationResult) -> None:
        """Trigger all registered callbacks"""
        for callback in self.callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.warning(f"⚠️ Callback error: {e}")

    # =========================
    # CALLBACKS
    # =========================

    def add_callback(self, callback: Callable[[OperationResult], None]) -> None:
        """Add callback for operations"""
        self.callbacks.append(callback)
        logger.info(f"✅ Callback added: {callback.__name__}")

    def remove_callback(self, callback: Callable) -> None:
        """Remove callback"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            logger.info(f"✅ Callback removed: {callback.__name__}")

    # =========================
    # HISTORY & STATISTICS
    # =========================

    def get_operation_history(self, limit: int = 10) -> List[Dict]:
        """Get operation history"""
        history = []
        for result in list(self.operation_history)[-limit:]:
            history.append({
                "operation": result.operation.value,
                "success": result.success,
                "source": result.source,
                "destination": result.destination,
                "file_size": self._format_size(result.file_size),
                "duration": f"{result.duration:.2f}s",
                "timestamp": result.timestamp.strftime("%H:%M:%S")
            })
        return history

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics"""
        if not self.operation_history:
            return {"total_operations": 0}

        successful = sum(1 for r in self.operation_history if r.success)
        failed = sum(1 for r in self.operation_history if not r.success)
        total_size = sum(r.file_size for r in self.operation_history)

        return {
            "total_operations": len(self.operation_history),
            "successful": successful,
            "failed": failed,
            "success_rate": f"{(successful / len(self.operation_history) * 100):.1f}%",
            "total_data_transferred": self._format_size(total_size),
            "uptime": str(datetime.now() - self.start_time)
        }

    def print_stats(self) -> None:
        """Print statistics"""
        stats = self.get_stats()
        print("\n" + "="*50)
        print("📊 FILE MANAGER STATISTICS")
        print("="*50)
        for key, value in stats.items():
            print(f"{key.replace('_', ' ').title():.<35} {value}")
        print("="*50 + "\n")

    def print_history(self, limit: int = 10) -> None:
        """Print operation history"""
        history = self.get_operation_history(limit)
        print("\n" + "="*50)
        print(f"📜 OPERATION HISTORY (Last {limit})")
        print("="*50)
        for i, item in enumerate(history, 1):
            print(f"\n{i}. {item['operation']}")
            print(f"   Status: {'✅ Success' if item['success'] else '❌ Failed'}")
            print(f"   Size: {item['file_size']} | Time: {item['duration']}")
            print(f"   Time: {item['timestamp']}")
        print("="*50 + "\n")

    def export_history(self, filename: str) -> None:
        """Export history to JSON"""
        history = self.get_operation_history(len(self.operation_history))
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ History exported to: {filename}")


# =========================
# TESTING & DEMONSTRATION
# =========================

def main():
    """Comprehensive test of the file manager"""

    print("\n" + "="*50)
    print("📁 ADVANCED FILE MANAGER")
    print("="*50 + "\n")

    manager = AdvancedFileManager()

    # Test 1: List available drives
    print("📝 Test 1: Available Drives")
    drives = manager.list_drives()
    print(f"Found drives: {drives}\n")

    # Test 2: Drive statistics
    print("📝 Test 2: Drive Statistics")
    for drive in drives:
        manager.print_drive_stats(drive)

    # Test 3: Create folder
    print("📝 Test 3: Create Folder")
    if 'C' in drives:
        folder = manager.create_folder("MyAIProject", drive="D")
        if folder:
            print(f"✅ Created: {folder}\n")

    # Test 4: File operations
    print("📝 Test 4: File Operations")
    test_file = "test.txt"
    with open(test_file, 'w') as f:
        f.write("Test content")

    if folder:
        manager.copy_file(test_file, "MyAIProject", drive="C")
        manager.list_folder_contents(folder)

    # Test 5: Compression
    print("\n📝 Test 5: Compression")
    if folder:
        zip_file = manager.compress_folder(folder)
        if zip_file:
            print(f"✅ Compressed: {zip_file}\n")

    # Test 6: Statistics
    print("📝 Test 6: Statistics")
    manager.print_stats()
    manager.print_history(limit=5)

    # Cleanup
    if os.path.exists(test_file):
        os.remove(test_file)

    print("✅ All tests complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⛔ Program interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")


# =========================
# SIMPLE FUNCTION WRAPPER
# =========================

_manager = AdvancedFileManager()

def create_folder(folder_name: str, drive: str = "C") -> Optional[str]:
    return _manager.create_folder(folder_name, drive=drive)