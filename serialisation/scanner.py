import os
import pickle
import io
import torch
import re
import sys
from typing import List, Tuple, Optional, Dict, Any, Set

class ModelScanner:
    """
    Scanner for detecting potential pickle-based serialization attacks in PyTorch model files.
    """
    
    SUSPICIOUS_MODULES = {
        'os', 'subprocess', 'sys', 'shutil', 'runpy', 'pty', 
        'socket', 'requests', 'urllib', 'ftplib'
    }
    
    SUSPICIOUS_FUNCTIONS = {
        'system', 'popen', 'exec', 'eval', 'execfile', 'compile',
        'open', 'rmtree', 'remove', 'unlink', 'chmod', 'chown'
    }
    
    SUSPICIOUS_PATTERNS = [
        r'(cat|grep|find).*\.(aws|ssh|conf|password|secret)',
        r'curl.*http',
        r'wget.*http',
        r'nc\s+-.*\d+',
        r'bash\s+-c',
        r'/bin/(sh|bash)',
        r'(>|>>)\s+/.*',
        r'rm\s+-rf',
        r'__reduce__',
        r'pickle_inject'
    ]

    def __init__(self, safe_mode=True):
        """
        Initialize the scanner.
        
        Args:
            safe_mode: If True, prevents execution of potentially malicious code
        """
        self.safe_mode = safe_mode
        self.findings = []
        self.detected_malicious = False
    
    def _safe_unpickler(self, data):
        """
        Create a safe unpickler that blocks potentially harmful operations.
        """
        class SafeUnpickler(pickle.Unpickler):
            def find_class(self_unpickler, module, name):
                raise pickle.UnpicklingError(
                    f"Blocked import of {module}.{name} for security reasons"
                )
        
        return SafeUnpickler(io.BytesIO(data))
    
    def _analyze_binary_content(self, data: bytes) -> List[str]:
        """
        Analyze binary content for suspicious patterns without unpickling.
        """
        findings = []
        
        try:
            data_str = data.decode('utf-8', errors='ignore')
        except:
            data_str = str(data)
        
        for module in self.SUSPICIOUS_MODULES:
            if module in data_str:
                findings.append(f"Suspicious module reference detected: {module}")
        
        for func in self.SUSPICIOUS_FUNCTIONS:
            if func in data_str:
                findings.append(f"Suspicious function call detected: {func}")
        
        for pattern in self.SUSPICIOUS_PATTERNS:
            matches = re.findall(pattern, data_str)
            if matches:
                findings.append(f"Suspicious pattern detected: {matches[0]}")
        
        if "PickleInject" in data_str:
            findings.append("PickleInject serialization attack pattern detected")
            
        return findings
    
    def _create_safe_torch_load(self):
        """
        Create a safe version of torch.load that won't execute malicious code.
        """
        original_torch_load = torch.load
        scanner = self
        
        def detect_only_torch_load(file_path, *args, **kwargs):
            """
            A replacement for torch.load that only detects malicious content
            without executing it.
            """
            with open(file_path, 'rb') as f:
                file_content = f.read()
            
            findings = scanner._analyze_binary_content(file_content)
            if findings:
                scanner.findings.extend(findings)
                scanner.detected_malicious = True
                return None
            
            if not scanner.safe_mode:
                kwargs['weights_only'] = True
                result = original_torch_load(file_path, *args, **kwargs)
                return result
            else:
                return None
        
        return detect_only_torch_load
    
    def scan_model_file(self, model_path: str) -> Tuple[bool, List[str]]:
        """
        Scan a PyTorch model file for possible serialization attacks.
        
        Args:
            model_path: Path to the .pt model file
            
        Returns:
            Tuple containing:
                - Boolean indicating if malicious content was detected
                - List of findings/details about the detected issues
        """
        self.findings = []
        self.detected_malicious = False
        
        if not os.path.exists(model_path):
            return False, ["File does not exist"]
        
        if not model_path.endswith('.pt') and not model_path.endswith('.pth'):
            return False, ["Not a PyTorch model file (.pt or .pth extension)"]
            
        try:
            with open(model_path, 'rb') as f:
                file_content = f.read()
            
            binary_findings = self._analyze_binary_content(file_content)
            if binary_findings:
                self.findings.extend(binary_findings)
                self.detected_malicious = True
                return True, self.findings
                
            original_torch_load = torch.load
            torch.load = self._create_safe_torch_load()
            
            try:
                torch.load(model_path, map_location='cpu')
            finally:
                torch.load = original_torch_load
            
            if self.detected_malicious:
                return True, self.findings
            
            return False, ["No malicious content detected"]
        
        except Exception as e:
            self.findings.append(f"Error during scanning: {str(e)}")
            return len(self.findings) > 1, self.findings
    
    def scan_directory(self, directory_path: str, recursive: bool = True) -> Dict[str, Tuple[bool, List[str]]]:
        """
        Scan all PyTorch model files in a directory.
        
        Args:
            directory_path: Path to the directory containing model files
            recursive: Whether to scan subdirectories
            
        Returns:
            Dictionary mapping file paths to scan results
        """
        results = {}
        
        if not os.path.isdir(directory_path):
            return {directory_path: (False, ["Not a directory"])}
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.pt') or file.endswith('.pth'):
                    file_path = os.path.join(root, file)
                    results[file_path] = self.scan_model_file(file_path)
            
            if not recursive:
                break
                
        return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Scan PyTorch model files for serialization attacks')
    parser.add_argument('path', help='Path to model file or directory')
    parser.add_argument('--recursive', '-r', action='store_true', help='Scan directories recursively')
    parser.add_argument('--unsafe', action='store_true', help='Allow loading of model content after scanning (potentially dangerous)')
    
    args = parser.parse_args()
    
    scanner = ModelScanner(safe_mode=not args.unsafe)
    
    if os.path.isdir(args.path):
        results = scanner.scan_directory(args.path, args.recursive)
        for path, (malicious, findings) in results.items():
            print(f"\nScanning: {path}")
            if malicious:
                print("⚠️ MALICIOUS CONTENT DETECTED!")
                for finding in findings:
                    print(f" - {finding}")
            else:
                print("✓ No malicious content detected")
    else:
        malicious, findings = scanner.scan_model_file(args.path)
        if malicious:
            print("⚠️ MALICIOUS CONTENT DETECTED!")
            for finding in findings:
                print(f" - {finding}")
        else:
            print("✓ No malicious content detected")