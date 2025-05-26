"""
Custom pylint checker for ML-specific code smells.
"""
import astroid
from pylint.checkers import BaseChecker


class MLChecker(BaseChecker):
    """Checker for ML-specific code smells."""
    
    name = 'ml-checker'
    priority = -1
    msgs = {
        'W9901': (
            'Use np.isnan() or pd.isna() instead of comparing with np.nan using == or !=',
            'nan-equivalence-misuse',
            'NaN values cannot be compared using == or != operators. '
            'Use np.isnan() for NumPy arrays or pd.isna()/pd.isnull() for Pandas data.'
        ),
    }
    
    def visit_compare(self, node):
        """Check for NaN equivalence comparison misuse."""
        try:
            # Check if left operand is NaN
            if self._is_nan_reference(node.left):
                self.add_message('nan-equivalence-misuse', node=node)
                return
                
            # Check ops list for comparisons involving NaN
            for op_name, comparator in node.ops:
                if self._is_nan_reference(comparator):
                    self.add_message('nan-equivalence-misuse', node=node)
                    break
        except Exception:
            # Silently ignore any parsing issues
            pass
    
    def _is_nan_reference(self, node):
        """Check if a node represents a reference to NaN."""
        if isinstance(node, astroid.Attribute):
            # Check for np.nan, numpy.nan, pd.nan, pandas.nan, etc.
            if node.attrname in ['nan', 'NaN', 'NAN']:
                return True
        elif isinstance(node, astroid.Name):
            # Check for direct references like 'nan' or 'NaN'
            if node.name in ['nan', 'NaN', 'NAN']:
                return True
        elif isinstance(node, astroid.Call):
            # Check for float('nan') calls
            if (isinstance(node.func, astroid.Name) and 
                node.func.name == 'float' and 
                len(node.args) == 1 and
                isinstance(node.args[0], astroid.Const) and
                node.args[0].value == 'nan'):
                return True
        return False


def register(linter):
    """Register the checker with pylint."""
    linter.register_checker(MLChecker(linter))
