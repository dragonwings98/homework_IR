from enum import Enum
from typing import List, Any


class TokenType(Enum):
    """Token type"""
    NUMBER = "NUMBER"
    STRING = "STRING"
    IDENTIFIER = "IDENTIFIER"
    KEYWORD = "KEYWORD"
    OPERATOR = "OPERATOR"
    DELIMITER = "DELIMITER"
    EOF = "EOF"


class Token:
    """Token"""

    def __init__(self, token_type: TokenType, value: str, line: int = 0):
        self.type = token_type
        self.value = value
        self.line = line

    def __repr__(self):
        return f"Token({self.type}, {self.value})"


class Lexer:
    """Lexer"""

    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.current_char = self.text[self.pos] if self.text else None
        self.line = 1

        self.keywords = {
            'print', 'read', 'if', 'else', 'while', 'for', 'let', 'def', 'return',
            'true', 'false', 'and', 'or', 'not', 'in', 'end'
        }

        self.operators = {
            '+', '-', '*', '/', '%', '=', '==', '!=', '<', '>', '<=', '>=',
            '&&', '||', '!', '+=', '-=', '*=', '/='
        }

        self.delimiters = {'(', ')', '[', ']', '{', '}', ',', ';', ':', '\n'}

    def advance(self):
        """Move to the next character"""
        if self.current_char == '\n':
            self.line += 1

        self.pos += 1
        if self.pos >= len(self.text):
            self.current_char = None
        else:
            self.current_char = self.text[self.pos]

    def skip_whitespace(self):
        """Skip whitespace characters"""
        while self.current_char and self.current_char in ' \t\r':
            self.advance()

    def skip_comment(self):
        """Skip comments"""
        if self.current_char == '#':
            while self.current_char and self.current_char != '\n':
                self.advance()

    def read_number(self) -> Token:
        """Read a number"""
        result = ''
        while self.current_char and (self.current_char.isdigit() or self.current_char == '.'):
            result += self.current_char
            self.advance()

        return Token(TokenType.NUMBER, result, self.line)

    def read_string(self) -> Token:
        """Read a string"""
        quote_char = self.current_char
        self.advance()

        result = ''
        while self.current_char and self.current_char != quote_char:
            if self.current_char == '\\':
                self.advance()
                if self.current_char == 'n':
                    result += '\n'
                elif self.current_char == 't':
                    result += '\t'
                elif self.current_char == '\\':
                    result += '\\'
                elif self.current_char == quote_char:
                    result += quote_char
                else:
                    result += self.current_char
            else:
                result += self.current_char
            self.advance()

        if self.current_char == quote_char:
            self.advance()

        return Token(TokenType.STRING, result, self.line)

    def read_identifier(self) -> Token:
        """Read an identifier or keyword"""
        result = ''
        while self.current_char and (self.current_char.isalnum() or self.current_char == '_'):
            result += self.current_char
            self.advance()

        token_type = TokenType.KEYWORD if result in self.keywords else TokenType.IDENTIFIER
        return Token(token_type, result, self.line)

    def read_operator(self) -> Token:
        """Read an operator"""
        # First check for two-character operators
        if self.current_char == '<' and self.peek() == '=':
            self.advance()
            self.advance()
            return Token(TokenType.OPERATOR, '<=', self.line)
        elif self.current_char == '>' and self.peek() == '=':
            self.advance()
            self.advance()
            return Token(TokenType.OPERATOR, '>=', self.line)
        elif self.current_char == '=' and self.peek() == '=':
            self.advance()
            self.advance()
            return Token(TokenType.OPERATOR, '==', self.line)
        elif self.current_char == '!' and self.peek() == '=':
            self.advance()
            self.advance()
            return Token(TokenType.OPERATOR, '!=', self.line)
        elif self.current_char == '&' and self.peek() == '&':
            self.advance()
            self.advance()
            return Token(TokenType.OPERATOR, '&&', self.line)
        elif self.current_char == '|' and self.peek() == '|':
            self.advance()
            self.advance()
            return Token(TokenType.OPERATOR, '||', self.line)

        # Then check for single-character operators
        char = self.current_char
        self.advance()
        return Token(TokenType.OPERATOR, char, self.line)

    def get_next_token(self) -> Token:
        """Get the next token"""
        while self.current_char:
            if self.current_char in ' \t\r':
                self.skip_whitespace()
                continue

            if self.current_char == '#':
                self.skip_comment()
                continue

            if self.current_char.isdigit():
                return self.read_number()

            if self.current_char in '"\'':
                return self.read_string()

            if self.current_char.isalpha() or self.current_char == '_':
                return self.read_identifier()

            if self.current_char in self.delimiters:
                char = self.current_char
                self.advance()
                return Token(TokenType.DELIMITER, char, self.line)

            # Handle operators
            if self.current_char in '+-*/%=<>!&|':
                return self.read_operator()

            char = self.current_char
            self.advance()
            raise SyntaxError(f"Unknown character '{char}' at line {self.line}")

        return Token(TokenType.EOF, '', self.line)

    def peek(self) -> str:
        """Peek at the next character without moving the position"""
        peek_pos = self.pos + 1
        if peek_pos >= len(self.text):
            return None
        return self.text[peek_pos]


class SimpleInterpreter:
    """Simple programming language interpreter"""

    def __init__(self):
        self.variables = {}
        self.functions = {}
        self.lexer = None
        self.current_token = None
        self.output_buffer = []

    def error(self, message: str):
        """Raise a syntax error"""
        line = self.current_token.line if self.current_token else 0
        raise SyntaxError(f"Line {line}: {message}")

    def eat(self, token_type: TokenType):
        """Consume a token of the specified type"""
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            self.error(f"Expected {token_type}, but got {self.current_token.type}")

    def parse_expression(self) -> Any:
        """Parse an expression"""
        return self.parse_or_expression()

    def parse_or_expression(self) -> Any:
        """Parse an OR expression"""
        result = self.parse_and_expression()

        while (self.current_token.type == TokenType.OPERATOR and
               self.current_token.value in ['or', '||']):
            op = self.current_token.value
            self.eat(TokenType.OPERATOR)
            right = self.parse_and_expression()

            if op == 'or':
                result = result or right
            elif op == '||':
                result = bool(result) or bool(right)

        return result

    def parse_and_expression(self) -> Any:
        """Parse an AND expression"""
        result = self.parse_equality_expression()

        while (self.current_token.type == TokenType.OPERATOR and
               self.current_token.value in ['and', '&&']):
            op = self.current_token.value
            self.eat(TokenType.OPERATOR)
            right = self.parse_equality_expression()

            if op == 'and':
                result = result and right
            elif op == '&&':
                result = bool(result) and bool(right)

        return result

    def parse_equality_expression(self) -> Any:
        """Parse an equality expression"""
        result = self.parse_relational_expression()

        while (self.current_token.type == TokenType.OPERATOR and
               self.current_token.value in ['==', '!=']):
            op = self.current_token.value
            self.eat(TokenType.OPERATOR)
            right = self.parse_relational_expression()

            if op == '==':
                result = result == right
            elif op == '!=':
                result = result != right

        return result

    def parse_relational_expression(self) -> Any:
        """Parse a relational expression"""
        result = self.parse_additive_expression()

        while (self.current_token.type == TokenType.OPERATOR and
               self.current_token.value in ['<', '>', '<=', '>=']):
            op = self.current_token.value
            self.eat(TokenType.OPERATOR)
            right = self.parse_additive_expression()

            if op == '<':
                result = result < right
            elif op == '>':
                result = result > right
            elif op == '<=':
                result = result <= right
            elif op == '>=':
                result = result >= right

        return result

    def parse_additive_expression(self) -> Any:
        """Parse an additive expression"""
        result = self.parse_multiplicative_expression()

        while (self.current_token.type == TokenType.OPERATOR and
               self.current_token.value in ['+', '-']):
            op = self.current_token.value
            self.eat(TokenType.OPERATOR)
            right = self.parse_multiplicative_expression()

            if op == '+':
                # When the operands contain strings, treat it as string concatenation
                if isinstance(result, str) or isinstance(right, str):
                    result = str(result) + str(right)
                else:
                    result = result + right
            elif op == '-':
                result = result - right

        return result

    def parse_multiplicative_expression(self) -> Any:
        """Parse a multiplicative expression"""
        result = self.parse_unary_expression()

        while (self.current_token.type == TokenType.OPERATOR and
               self.current_token.value in ['*', '/', '%']):
            op = self.current_token.value
            self.eat(TokenType.OPERATOR)
            right = self.parse_unary_expression()

            if op == '*':
                result = result * right
            elif op == '/':
                if right == 0:
                    self.error("Division by zero error")
                result = result / right
            elif op == '%':
                result = result % right

        return result

    def parse_unary_expression(self) -> Any:
        """Parse a unary expression"""
        if self.current_token.type == TokenType.OPERATOR and self.current_token.value in ['+', '-', 'not']:
            op = self.current_token.value
            self.eat(TokenType.OPERATOR)
            expr = self.parse_unary_expression()

            if op == '+':
                return +expr
            elif op == '-':
                return -expr
            elif op == 'not':
                return not expr

        return self.parse_primary_expression()

    def parse_primary_expression(self) -> Any:
        """Parse a primary expression"""
        if self.current_token.type == TokenType.NUMBER:
            value = self.current_token.value
            self.eat(TokenType.NUMBER)
            return float(value) if '.' in value else int(value)

        elif self.current_token.type == TokenType.STRING:
            value = self.current_token.value
            self.eat(TokenType.STRING)
            return value

        elif self.current_token.type == TokenType.KEYWORD and self.current_token.value in ['true', 'false']:
            value = self.current_token.value == 'true'
            self.eat(TokenType.KEYWORD)
            return value

        elif self.current_token.type == TokenType.IDENTIFIER:
            var_name = self.current_token.value
            self.eat(TokenType.IDENTIFIER)

            if var_name not in self.variables:
                self.error(f"Undefined variable '{var_name}'")

            return self.variables[var_name]

        elif self.current_token.type == TokenType.DELIMITER and self.current_token.value == '(':
            self.eat(TokenType.DELIMITER)
            result = self.parse_expression()
            self.eat(TokenType.DELIMITER)
            return result

        else:
            self.error(f"Unexpected token {self.current_token}")

    def execute_statement(self):
        """Execute a statement"""
        if self.current_token.type == TokenType.KEYWORD:
            if self.current_token.value == 'let':
                self.execute_assignment()
            elif self.current_token.value == 'print':
                self.execute_print()
            elif self.current_token.value == 'read':
                self.execute_read()
            elif self.current_token.value == 'if':
                self.execute_if()
            elif self.current_token.value == 'while':
                self.execute_while()
            else:
                self.error(f"Unknown keyword '{self.current_token.value}'")

        elif self.current_token.type == TokenType.IDENTIFIER:
            var_name = self.current_token.value
            self.eat(TokenType.IDENTIFIER)

            if self.current_token.type == TokenType.OPERATOR and self.current_token.value == '=':
                self.eat(TokenType.OPERATOR)
                value = self.parse_expression()
                self.variables[var_name] = value
            else:
                self.error("Expected assignment operator '='")

    def execute_assignment(self):
        """Execute variable assignment"""
        self.eat(TokenType.KEYWORD)

        if self.current_token.type != TokenType.IDENTIFIER:
            self.error("Expected variable name")

        var_name = self.current_token.value
        self.eat(TokenType.IDENTIFIER)
        self.eat(TokenType.OPERATOR)

        value = self.parse_expression()
        self.variables[var_name] = value

    def execute_print(self):
        """Execute a print statement"""
        self.eat(TokenType.KEYWORD)

        if self.current_token.type == TokenType.DELIMITER and self.current_token.value == '(':
            self.eat(TokenType.DELIMITER)
            value = self.parse_expression()
            self.eat(TokenType.DELIMITER)
        else:
            value = self.parse_expression()

        output = str(value)
        print(output)
        self.output_buffer.append(output)

    def execute_read(self):
        """Execute a read statement"""
        self.eat(TokenType.KEYWORD)

        if self.current_token.type != TokenType.IDENTIFIER:
            self.error("Expected variable name")

        var_name = self.current_token.value
        self.eat(TokenType.IDENTIFIER)

        try:
            user_input = input(f"Please enter {var_name}: ")
            try:
                value = float(user_input) if '.' in user_input else int(user_input)
            except ValueError:
                value = user_input

            self.variables[var_name] = value
        except EOFError:
            self.variables[var_name] = ""

    def execute_if(self):
        """Execute a conditional statement"""
        self.eat(TokenType.KEYWORD)  # Consume 'if'

        condition = self.parse_expression()

        # Handle curly braces or newlines
        has_braces = False
        if self.current_token.type == TokenType.DELIMITER:
            if self.current_token.value == '{':
                has_braces = True
                self.eat(TokenType.DELIMITER)  # Consume '{'
            elif self.current_token.value == '\n':
                self.eat(TokenType.DELIMITER)  # Consume newline

        # Execute the code block when the condition is true
        if condition:
            while True:
                if self.current_token.type == TokenType.EOF:
                    self.error("Incomplete if statement")

                # Check if it's else or end
                if (self.current_token.type == TokenType.KEYWORD and
                        self.current_token.value in ['else', 'end']):
                    break

                # Check if it's a closing curly brace
                if has_braces and (self.current_token.type == TokenType.DELIMITER and
                                   self.current_token.value == '}'):
                    self.eat(TokenType.DELIMITER)  # Consume '}'
                    break

                self.execute_statement()
                self.skip_newlines()
        else:
            # Skip the code block when the condition is false
            self.skip_block(['else', 'end', '}'] if has_braces else ['else', 'end'])

        # Handle the else branch
        if (self.current_token.type == TokenType.KEYWORD and
                self.current_token.value == 'else'):
            self.eat(TokenType.KEYWORD)  # Consume 'else'

            # Check if it's else if
            if (self.current_token.type == TokenType.KEYWORD and
                    self.current_token.value == 'if'):
                self.execute_if()  # Recursively handle else if
                return

            # Handle the else code block
            has_else_braces = False
            if self.current_token.type == TokenType.DELIMITER:
                if self.current_token.value == '{':
                    has_else_braces = True
                    self.eat(TokenType.DELIMITER)  # Consume '{'
                elif self.current_token.value == '\n':
                    self.eat(TokenType.DELIMITER)  # Consume newline

            if not condition:
                # Execute the else code block
                while True:
                    if self.current_token.type == TokenType.EOF:
                        self.error("Incomplete else statement")

                    # Check if it's end
                    if (self.current_token.type == TokenType.KEYWORD and
                            self.current_token.value == 'end'):
                        break

                    # Check if it's a closing curly brace
                    if has_else_braces and (self.current_token.type == TokenType.DELIMITER and
                                            self.current_token.value == '}'):
                        self.eat(TokenType.DELIMITER)  # Consume '}'
                        break

                    self.execute_statement()
                    self.skip_newlines()
            else:
                # Skip the else code block
                self.skip_block(['end', '}'] if has_else_braces else ['end'])

        # Consume the end marker
        if not has_braces:
            if (self.current_token.type == TokenType.KEYWORD and
                    self.current_token.value == 'end'):
                self.eat(TokenType.KEYWORD)  # Consume 'end'

    def execute_while(self):
        """Execute a loop statement"""
        self.eat(TokenType.KEYWORD)  # Consume 'while'

        loop_start_pos = self.lexer.pos
        loop_start_char = self.lexer.current_char
        loop_start_line = self.lexer.line

        # Handle curly braces or newlines
        has_braces = False
        if self.current_token.type == TokenType.DELIMITER:
            if self.current_token.value == '{':
                has_braces = True
                self.eat(TokenType.DELIMITER)  # Consume '{'
            elif self.current_token.value == '\n':
                self.eat(TokenType.DELIMITER)  # Consume newline

        while True:
            # Reset the lexer to the start of the loop
            self.lexer.pos = loop_start_pos
            self.lexer.current_char = loop_start_char
            self.lexer.line = loop_start_line
            self.current_token = self.lexer.get_next_token()

            # Skip the curly brace or newline after the loop condition
            if has_braces:
                self.eat(TokenType.DELIMITER)  # Consume '{'
            else:
                self.skip_newlines()

            condition = self.parse_expression()

            if not condition:
                break

            # Execute the loop body
            while True:
                if self.current_token.type == TokenType.EOF:
                    self.error("Incomplete while statement")

                # Check if it's end
                if (self.current_token.type == TokenType.KEYWORD and
                        self.current_token.value == 'end'):
                    break

                # Check if it's a closing curly brace
                if has_braces and (self.current_token.type == TokenType.DELIMITER and
                                   self.current_token.value == '}'):
                    self.eat(TokenType.DELIMITER)  # Consume '}'
                    break

                self.execute_statement()
                self.skip_newlines()

        # Consume the end marker
        if not has_braces:
            if (self.current_token.type == TokenType.KEYWORD and
                    self.current_token.value == 'end'):
                self.eat(TokenType.KEYWORD)  # Consume 'end'

    def skip_newlines(self):
        """Skip newline characters"""
        while (self.current_token.type == TokenType.DELIMITER and
               self.current_token.value == '\n'):
            self.eat(TokenType.DELIMITER)

    def skip_block(self, end_keywords: List[str]):
        """Skip a code block"""
        depth = 1
        while depth > 0 and self.current_token.type != TokenType.EOF:
            if self.current_token.type == TokenType.KEYWORD:
                if self.current_token.value in ['if', 'while']:
                    depth += 1
                elif self.current_token.value in end_keywords:
                    depth -= 1
            elif self.current_token.type == TokenType.DELIMITER:
                if self.current_token.value == '{':
                    depth += 1
                elif self.current_token.value == '}':
                    depth -= 1
            self.current_token = self.lexer.get_next_token()

    def interpret(self, text: str) -> List[str]:
        """Interpret and execute the code"""
        self.lexer = Lexer(text)
        self.current_token = self.lexer.get_next_token()
        self.output_buffer = []

        try:
            while self.current_token.type != TokenType.EOF:
                if (self.current_token.type == TokenType.DELIMITER and
                        self.current_token.value == '\n'):
                    self.eat(TokenType.DELIMITER)
                    continue

                self.execute_statement()
                self.skip_newlines()

            return self.output_buffer

        except Exception as e:
            print(f"Runtime error: {e}")
            return self.output_buffer


def run_demo():
    """Run the demo program"""
    interpreter = SimpleInterpreter()

    print("=== Simple Programming Language Interpreter Demo ===")
    print("The following shows several example programs to demonstrate the basic functions of the interpreter\n")

    # Example 1: Basic mathematical operations and printing
    print("Example 1: Basic mathematical operations and printing")
    code1 = """
    let a = 10
    let b = 3
    print a + b  # Addition
    print a - b  # Subtraction
    print a * b  # Multiplication
    print a / b  # Division
    print a % b  # Modulus
    """
    print("Code:")
    print(code1)
    print("Execution result:")
    interpreter.interpret(code1)
    print()

    # Example 2: Variable assignment and string operations
    print("Example 2: Variable assignment and string operations")
    code2 = """
    let name = "World"
    print "Hello, " + name
    let num = 42
    print "The answer is: " + num
    """
    print("Code:")
    print(code2)
    print("Execution result:")
    interpreter.interpret(code2)
    print()

    # Example 3: Conditional statements
    print("Example 3: Conditional statements")
    code3 = """
    let x = 15
    if x > 10 {
        print "x is greater than 10"
    } else {
        print "x is less than or equal to 10"
    }

    let y = 5
    if y > 10 {
        print "y is greater than 10"
    } else if y < 10 {
        print "y is less than 10"
    } else {
        print "y is equal to 10"
    }
    """
    print("Code:")
    print(code3)
    print("Execution result:")
    interpreter.interpret(code3)
    print()

# Run the demo program
if __name__ == "__main__":
    run_demo()