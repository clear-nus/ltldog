from typing import Any, List
import pdb

class LTLParseError(Exception):
    """Custom exception for LTL parsing errors."""
    pass

class LTLParser:
    def __init__(self, propositions:List[str]):
        self.props = list(set(list(propositions+['True', 'False', '1', '0'])))
        self.operators = {'&': 'and', '|': 'or', '~': 'not', '!': 'not', 'X': 'next', 'U': 'until', 'F': 'eventually', 'G': 'always'}
        self.operators = {'&': 'and', '|': 'or', '~': 'not', '!': 'not', 'X': 'next', 'U': 'until', 'G': 'always'}
        self.precedence = {'~': 3, '!': 3, 'X': 3, 'F': 3, 'G': 3, '&': 2, '|': 2, 'U': 1}


    def __call__(self, *args, **kwargs):
        return self.parse(*args, **kwargs)


    def reverse_infix(self, infix):
        """Reverses the infix expression for conversion."""
        reversed_infix = []
        for token in reversed(infix):
            if token == '(':
                reversed_infix.append(')')
            elif token == ')':
                reversed_infix.append('(')
            else:
                reversed_infix.append(token)
        #end for
        return reversed_infix

    def reversed_infix_to_postfix(self, reversed_infix):
        """Converts a reversed infix expression to a postfix expression."""
        stack = []
        postfix = []
        for token in reversed_infix:
            if token in self.props:
                postfix.append(token)
            elif token in self.operators:
                while stack and self.precedence.get(stack[-1], 0) >= self.precedence.get(token, 0):
                    postfix.append(stack.pop())
                stack.append(token)
            elif token == '(':
                stack.append(token)
            elif token == ')':
                while stack and stack[-1] != '(':
                    postfix.append(stack.pop())
                stack.pop()  # Pop '(' from stack
        #end for
        while stack:
            postfix.append(stack.pop())
        #end while
        
        return postfix


    def tokenize(self, rule_str):
        """Converts the rule string into a list of tokens."""
        tokens = []
        i = 0
        while i < len(rule_str):
            if rule_str[i] in self.operators:
                tokens.append(rule_str[i])
                i += 1
            elif rule_str[i] == '(' or rule_str[i] == ')':
                tokens.append(rule_str[i])
                i += 1
            elif rule_str[i].isspace():
                i += 1
            else:
                # Propositions
                start = i
                while i < len(rule_str) and rule_str[i].isalnum():
                    i += 1
                token = rule_str[start:i]
                if token in self.props:
                    tokens.append(token)
                else:
                    raise LTLParseError(f"Undeclared proposition \'{token}\'!")
        #end while
        
        return tokens


    def postfix_to_tuple(self, postfix_expression):
        """Converts a postfix expression to a nested tuple."""
        stack = []
        # pdb.set_trace()
        for token in postfix_expression:
            if token in self.operators:
                # Convert operator symbols to names
                token = self.operators[token]
                if token in ['and', 'or', 'until']:
                    # These operators require two operands
                    if len(stack) < 2:
                        raise LTLParseError("Insufficient operands for binary operator")
                    operand1 = stack.pop()
                    operand2 = stack.pop()
                    stack.append((token, operand1, operand2))
                elif token in ['not', 'next', 'eventually', 'always']:
                    # These operators require one operand
                    if not stack:
                        raise LTLParseError("Insufficient operands for unary operator")
                    operand = stack.pop()
                    stack.append((token, operand))
            else:
                # Propositions
                stack.append(token)

        if len(stack) != 1:
            raise LTLParseError("Invalid expression")
        return stack.pop()


    def parse(self, rule_str):
        infix = self.tokenize(rule_str)
        reversed_infix = self.reverse_infix(infix)
        postfix = self.reversed_infix_to_postfix(reversed_infix)
        # prefix = list(reversed(postfix))
        tup = self.postfix_to_tuple(postfix)

        return tup


def test_single():
    props = ['p1', 'p2', 'p3']
    parser = LTLParser(props)

    ltls = ["(~p2 U p1)"]
    # rule_str = 'G ( (p1 & p2) U ~p3 )'
    # # rule_str = "(p1 & p2)"

    try:
        tup = parser.parse(ltls[0])
    except LTLParseError as e:
        tup = str(e)

    print(tup)


def test():
    from tqdm import tqdm
    # file_path = 'LTL_dataset/3_10/formula.txt'
    file_path = 'LTL_dataset/6_20/formula.txt'
    file_path = 'datasets/ltls.txt'
    with open(file_path, 'r') as file:
        formulas = file.readlines()

    print(f"Read in total {len(formulas)} rules.")
    props = ['p0', 'p1', 'p2']
    parser = LTLParser(props)

    success = 0
    fail = 0
    for idx, rule_str in enumerate(tqdm(formulas)):
        try:
            tup = parser.parse(rule_str)
            print(tup)
        except LTLParseError as e:
            err_msg = str(e)
            fail += 1
            print(f"Case {idx} test failed!")
            print(f"rule_str = {rule_str}")
            print(f"Exception encountered: {err_msg}\n")
            continue
        success += 1
        # print(f'{idx} success: rule = {rule_str}')
    
    print(f"Success rate = {success/len(formulas)}")


if __name__ == '__main__':
    # test_single()
    test()
