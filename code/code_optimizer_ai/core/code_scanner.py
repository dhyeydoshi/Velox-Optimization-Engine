import ast
import asyncio
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from code.code_optimizer_ai.config.settings import settings
from code.code_optimizer_ai.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CodeFile:
    file_path: str
    content: str
    ast_tree: ast.AST
    file_hash: str
    last_modified: datetime
    size_bytes: int
    functions: List[str]
    classes: List[str]
    imports: List[str]
    complexity_metrics: Dict[str, float]


@dataclass
class ScanResult:
    scanned_files: int
    total_size_bytes: int
    files_by_type: Dict[str, int]
    scan_duration: float
    errors: List[str]
    code_files: List[CodeFile]


class CodeScanner:

    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    IGNORE_DIRS = {
        '__pycache__', '.git', '.pytest_cache', 'venv', 'env',
        'node_modules', '.tox', '.mypy_cache', '.coverage'
    }

    def __init__(self):
        pass

    async def scan_directory(self, directory_path: str, recursive: bool = True) -> ScanResult:
        return await asyncio.to_thread(self._scan_directory_sync, directory_path, recursive)

    def _scan_directory_sync(self, directory_path: str, recursive: bool) -> ScanResult:
        start = datetime.now()
        directory = Path(directory_path).expanduser().resolve()

        if not directory.exists() or not directory.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        files, total_size, errors = [], 0, []
        pattern = "**/*.py" if recursive else "*.py"

        for file_path in directory.glob(pattern):
            if len(files) >= settings.MAX_SCAN_FILES:
                errors.append(
                    f"Scan aborted after reaching MAX_SCAN_FILES={settings.MAX_SCAN_FILES}"
                )
                break

            if file_path.is_symlink():
                continue

            if self._should_ignore_file(file_path):
                continue

            try:
                try:
                    file_size = file_path.stat().st_size
                except OSError:
                    continue
                if total_size + file_size > settings.MAX_SCAN_TOTAL_BYTES:
                    errors.append(
                        f"Scan aborted after reaching MAX_SCAN_TOTAL_BYTES={settings.MAX_SCAN_TOTAL_BYTES}"
                    )
                    break

                code_file = self._analyze_file(file_path)
                if code_file:
                    files.append(code_file)
                    total_size += code_file.size_bytes
            except Exception as e:
                error_msg = f"Error analyzing {file_path}: {e}"
                errors.append(error_msg)
                logger.warning(error_msg)

        return ScanResult(
            scanned_files=len(files),
            total_size_bytes=total_size,
            files_by_type={"python": len(files)},
            scan_duration=(datetime.now() - start).total_seconds(),
            errors=errors,
            code_files=files
        )
    
    async def scan_file(self, file_path: str) -> Optional[CodeFile]:
        path = Path(file_path)
        return await asyncio.to_thread(self._analyze_file, path)
    
    def _analyze_file(self, file_path: Path) -> Optional[CodeFile]:
        try:
            if file_path.stat().st_size > self.MAX_FILE_SIZE:
                logger.warning(f"Skipping large file: {file_path}")
                return None
            
            content = file_path.read_text(encoding='utf-8')
            
            try:
                ast_tree = ast.parse(content, filename=str(file_path))
            except SyntaxError as e:
                logger.warning(f"Syntax error in {file_path}: {e}")
                return None
            
            return CodeFile(
                file_path=str(file_path),
                content=content,
                ast_tree=ast_tree,
                file_hash=hashlib.sha256(content.encode('utf-8')).hexdigest(),
                last_modified=datetime.fromtimestamp(file_path.stat().st_mtime),
                size_bytes=len(content.encode('utf-8')),
                functions=self._extract_functions(ast_tree),
                classes=self._extract_classes(ast_tree),
                imports=self._extract_imports(ast_tree),
                complexity_metrics=self._calculate_complexity_metrics(ast_tree)
            )
        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")
            return None
    
    def _should_ignore_file(self, file_path: Path) -> bool:
        # Check for ignored directories
        if any(ignore_dir in file_path.parts for ignore_dir in self.IGNORE_DIRS):
            return True

        # Only process .py files
        if file_path.suffix.lower() != '.py':
            return True
        
        # Skip test files and common setup scripts
        skip_names = {'test_', 'conftest', 'setup', 'manage'}
        if any(file_path.stem.startswith(name) for name in skip_names):
            return True
        
        return False
    
    def _extract_functions(self, ast_tree: ast.AST) -> List[str]:
        return [node.name for node in ast.walk(ast_tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]

    def _extract_classes(self, ast_tree: ast.AST) -> List[str]:
        return [node.name for node in ast.walk(ast_tree)
                if isinstance(node, ast.ClassDef)]

    def _extract_imports(self, ast_tree: ast.AST) -> List[str]:
        imports = []
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports.extend(
                    f"{module}.{alias.name}" if module else alias.name
                    for alias in node.names
                )
        return imports
    
    def _calculate_complexity_metrics(self, ast_tree: ast.AST) -> Dict[str, float]:
        try:
            cyclomatic = self._calculate_cyclomatic_complexity(ast_tree)
            halstead = self._calculate_halstead_complexity(ast_tree)
            loc = len(ast.unparse(ast_tree).split('\n'))

            return {
                'cyclomatic_complexity': cyclomatic,
                'halstead_complexity': halstead,
                'maintainability_index': self._calculate_maintainability_index(
                    cyclomatic, halstead, loc
                ),
                'lines_of_code': loc,
                'comment_lines': 0,
                'blank_lines': 0
            }
        except Exception as e:
            logger.warning(f"Error calculating metrics: {e}")
            return {k: 0.0 for k in ['cyclomatic_complexity', 'halstead_complexity',
                                      'maintainability_index', 'lines_of_code',
                                      'comment_lines', 'blank_lines']}

    def _calculate_cyclomatic_complexity(self, ast_tree: ast.AST) -> float:
        complexity = 1
        for node in ast.walk(ast_tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor,
                               ast.ExceptHandler, ast.And, ast.Or, ast.comprehension)):
                complexity += 1
        return float(complexity)
    
    def _calculate_halstead_complexity(self, ast_tree: ast.AST) -> float:
        operators, operands = set(), set()

        for node in ast.walk(ast_tree):
            if isinstance(node, (ast.BinOp, ast.UnaryOp)):
                operators.add(type(node.op).__name__)
            elif isinstance(node, ast.Compare):
                operators.update(type(op).__name__ for op in node.ops)
            elif isinstance(node, ast.Name):
                operands.add(node.id)
            elif isinstance(node, ast.Constant):
                operands.add(str(node.value))
        
        if not operators or not operands:
            return 0.0
        
        return (len(operators) / 2) * (len(operands) / 2)

    def _calculate_maintainability_index(self, cyclomatic: float,
                                         halstead: float, loc: float) -> float:
        if loc == 0:
            return 0.0
        
        complexity_factor = min(cyclomatic / 10, 1.0)
        halstead_factor = min(halstead / 100, 1.0)
        size_factor = min(loc / 500, 1.0)
        
        score = 100 - (complexity_factor * 30 + halstead_factor * 30 + size_factor * 30)
        return max(0, score)

    async def get_code_snippets(
        self,
        code_files: List[CodeFile],
        max_snippet_length: int = 1000
    ) -> List[Tuple[str, str, str]]:
        
        snippets = []
        
        for code_file in code_files:
            # Split code into logical chunks (functions, classes, modules)
            try:
                chunks = self._split_code_into_chunks(code_file, max_snippet_length)
                
                for i, chunk in enumerate(chunks):
                    identifier = f"{code_file.file_path}:chunk_{i}"
                    snippets.append((chunk, code_file.file_path, identifier))
                    
            except Exception as e:
                logger.warning(f"Failed to split code file {code_file.file_path}: {e}")
        
        return snippets
    
    def _split_code_into_chunks(
        self,
        code_file: CodeFile,
        max_length: int
    ) -> List[str]:
        
        content = code_file.content
        chunks = []
        
        # If file is small enough, analyze as single chunk
        if len(content) <= max_length:
            return [content]
        
        # Split by line-count / max character length
        lines = content.split('\n')
        current_chunk = []
        current_length = 0
        
        for line in lines:
            line_length = len(line) + 1  # +1 for newline
            
            # Check if adding this line would exceed max length
            if current_length + line_length > max_length and current_chunk:
                # Finalize current chunk
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_length = line_length
            else:
                current_chunk.append(line)
                current_length += line_length
        
        # Add final chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def get_scan_statistics(self, scan_result: ScanResult) -> Dict[str, Any]:
        
        if not scan_result.code_files:
            return {"error": "No code files found"}
        
        # Calculate aggregate statistics
        total_functions = sum(len(f.functions) for f in scan_result.code_files)
        total_classes = sum(len(f.classes) for f in scan_result.code_files)
        total_imports = sum(len(f.imports) for f in scan_result.code_files)
        
        # Calculate average complexity
        avg_cyclomatic = sum(
            f.complexity_metrics.get('cyclomatic_complexity', 0)
            for f in scan_result.code_files
        ) / len(scan_result.code_files)
        
        avg_maintainability = sum(
            f.complexity_metrics.get('maintainability_index', 0)
            for f in scan_result.code_files
        ) / len(scan_result.code_files)
        
        # Identify potential optimization targets
        high_complexity_files = [
            f.file_path for f in scan_result.code_files
            if f.complexity_metrics.get('cyclomatic_complexity', 0) > 10
        ][:10]
        
        low_maintainability_files = [
            f.file_path for f in scan_result.code_files
            if f.complexity_metrics.get('maintainability_index', 0) < 65
        ][:10]

        performance_candidates = [
            f.file_path for f in scan_result.code_files
            if (
                len(f.functions) > 5
                or len(f.classes) > 3
                or "compute" in f.file_path.lower()
                or "process" in f.file_path.lower()
            )
        ][:10]
        
        return {
            "total_files": scan_result.scanned_files,
            "total_size_mb": scan_result.total_size_bytes / (1024 * 1024),
            "total_functions": total_functions,
            "total_classes": total_classes,
            "total_imports": total_imports,
            "average_cyclomatic_complexity": round(avg_cyclomatic, 2),
            "average_maintainability_index": round(avg_maintainability, 2),
            "high_complexity_files": len(high_complexity_files),
            "high_complexity_file_paths": high_complexity_files,
            "low_maintainability_files": len(low_maintainability_files),
            "low_maintainability_file_paths": low_maintainability_files,
            "performance_candidate_files": performance_candidates,
            "scan_duration_seconds": round(scan_result.scan_duration, 2),
            "files_by_type": scan_result.files_by_type,
            "errors_count": len(scan_result.errors)
        }


# Global instance
code_scanner = CodeScanner()
