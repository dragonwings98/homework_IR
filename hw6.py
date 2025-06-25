from typing import Dict, Set, Tuple, List, Optional, FrozenSet
import re

class NFAState:
    """NFA状态类"""

    def __init__(self, state_id: int, is_final: bool = False):
        self.state_id = state_id
        self.is_final = is_final
        # 使用默认字典避免KeyError
        self.transitions: Dict[str, Set[int]] = {}  # 字符到状态的转换
        self.epsilon_transitions: Set[int] = set()  # ε转换


class DFAState:
    """DFA状态类"""

    def __init__(self, state_id: int, nfa_states: Set[int], is_final: bool = False):
        self.state_id = state_id
        self.nfa_states = nfa_states  # 对应的NFA状态集合
        self.is_final = is_final
        self.transitions: Dict[str, int] = {}  # 字符到状态的转换


class RegexDFA:
    """Regular expression DFA class"""

    def __init__(self, regex_pattern: str):
        self.regex_pattern = regex_pattern
        self.nfa_states = {}
        self.dfa_states = {}
        self.start_state = 0
        self.current_state_id = 0
        self.alphabet = set()

        self._build_dfa()

    def _get_next_state_id(self) -> int:
        """Get the next status ID"""
        state_id = self.current_state_id
        self.current_state_id += 1
        return state_id

    def _build_nfa_from_regex(self, pattern: str) -> Tuple[int, int]:
        """Build NFA from Regular Expression"""
        if not pattern:
            start = self._get_next_state_id()
            end = self._get_next_state_id()
            self.nfa_states[start] = NFAState(start)
            self.nfa_states[end] = NFAState(end, is_final=True)
            self.nfa_states[start].epsilon_transitions.add(end)
            return start, end

        return self._parse_regex(pattern)

    def _parse_regex(self, pattern: str) -> Tuple[int, int]:
        """Parse regular expressions"""
        if len(pattern) == 1:
            return self._create_char_nfa(pattern)

        # 处理括号分组
        if pattern.startswith('(') and self._find_matching_paren(pattern, 0) == len(pattern) - 1:
            inner_pattern = pattern[1:-1]
            return self._parse_regex(inner_pattern)

        # 处理或操作
        paren_balance = 0
        for i in range(len(pattern) - 1, -1, -1):
            if pattern[i] == ')':
                paren_balance += 1
            elif pattern[i] == '(':
                paren_balance -= 1
            elif pattern[i] == '|' and paren_balance == 0:
                left = pattern[:i]
                right = pattern[i + 1:]
                return self._create_alternation_nfa(left, right)

        # 处理Kleene星、加号和问号
        if pattern.endswith('*'):
            base_pattern = pattern[:-1]
            # 处理括号情况
            if base_pattern.endswith(')') and base_pattern.startswith('('):
                base_pattern = base_pattern[1:-1]
            base_start, base_end = self._parse_regex(base_pattern)
            return self._create_kleene_star_nfa(base_start, base_end)

        if pattern.endswith('+'):
            base_pattern = pattern[:-1]
            # 处理括号情况
            if base_pattern.endswith(')') and base_pattern.startswith('('):
                base_pattern = base_pattern[1:-1]
            base_start, base_end = self._parse_regex(base_pattern)
            return self._create_plus_nfa(base_start, base_end)

        if pattern.endswith('?'):
            base_pattern = pattern[:-1]
            # 处理括号情况
            if base_pattern.endswith(')') and base_pattern.startswith('('):
                base_pattern = base_pattern[1:-1]
            base_start, base_end = self._parse_regex(base_pattern)
            return self._create_optional_nfa(base_start, base_end)

        # 处理连接
        return self._create_concatenation_nfa(pattern)

    def _find_matching_paren(self, pattern: str, start: int) -> int:
        """查找匹配的右括号位置"""
        if pattern[start] != '(':
            return -1
        balance = 1
        for i in range(start + 1, len(pattern)):
            if pattern[i] == '(':
                balance += 1
            elif pattern[i] == ')':
                balance -= 1
                if balance == 0:
                    return i
        return -1

    def _create_char_nfa(self, char: str) -> Tuple[int, int]:
        """Create single-character NFA"""
        start = self._get_next_state_id()
        end = self._get_next_state_id()

        self.nfa_states[start] = NFAState(start)
        self.nfa_states[end] = NFAState(end, is_final=True)

        # 使用安全的方式添加转换
        self._add_transition(start, char, end)
        self.alphabet.add(char)

        return start, end

    def _add_transition(self, state_id: int, char: str, target_id: int) -> None:
        """安全地添加状态转换"""
        if state_id not in self.nfa_states:
            self.nfa_states[state_id] = NFAState(state_id)

        # 确保转换字典已初始化
        if char not in self.nfa_states[state_id].transitions:
            self.nfa_states[state_id].transitions[char] = set()

        # 添加转换
        self.nfa_states[state_id].transitions[char].add(target_id)

    def _create_concatenation_nfa(self, pattern: str) -> Tuple[int, int]:
        """Create a connected NFA"""
        if len(pattern) == 1:
            return self._create_char_nfa(pattern)

        # 寻找最安全的分割点（不在括号内的位置）
        split_point = self._find_safe_split(pattern)
        if split_point != -1:
            left = pattern[:split_point]
            right = pattern[split_point:]
        else:
            # 如果没有找到安全分割点，默认分割第一个字符
            left = pattern[0]
            right = pattern[1:]

        left_start, left_end = self._parse_regex(left)
        right_start, right_end = self._parse_regex(right)

        self.nfa_states[left_end].is_final = False
        self.nfa_states[left_end].epsilon_transitions.add(right_start)

        return left_start, right_end

    def _find_safe_split(self, pattern: str) -> int:
        """寻找连接操作的安全分割点"""
        paren_balance = 0
        for i in range(len(pattern) - 1, 0, -1):
            if pattern[i] == ')':
                paren_balance += 1
            elif pattern[i] == '(':
                paren_balance -= 1
            elif paren_balance == 0:
                # 检查是否不是操作符后面
                if pattern[i] not in '*+?|':
                    return i
        return -1

    def _create_alternation_nfa(self, left_pattern: str, right_pattern: str) -> Tuple[int, int]:
        """Create alternation NFA"""
        left_start, left_end = self._parse_regex(left_pattern)
        right_start, right_end = self._parse_regex(right_pattern)

        new_start = self._get_next_state_id()
        new_end = self._get_next_state_id()

        self.nfa_states[new_start] = NFAState(new_start)
        self.nfa_states[new_end] = NFAState(new_end, is_final=True)

        self.nfa_states[new_start].epsilon_transitions.add(left_start)
        self.nfa_states[new_start].epsilon_transitions.add(right_start)

        self.nfa_states[left_end].is_final = False
        self.nfa_states[right_end].is_final = False
        self.nfa_states[left_end].epsilon_transitions.add(new_end)
        self.nfa_states[right_end].epsilon_transitions.add(new_end)

        return new_start, new_end

    def _create_kleene_star_nfa(self, start: int, end: int) -> Tuple[int, int]:
        """Create Kleene star NFA"""
        new_start = self._get_next_state_id()
        new_end = self._get_next_state_id()

        self.nfa_states[new_start] = NFAState(new_start)
        self.nfa_states[new_end] = NFAState(new_end, is_final=True)

        self.nfa_states[new_start].epsilon_transitions.add(start)
        self.nfa_states[new_start].epsilon_transitions.add(new_end)

        self.nfa_states[end].is_final = False
        self.nfa_states[end].epsilon_transitions.add(start)
        self.nfa_states[end].epsilon_transitions.add(new_end)

        return new_start, new_end

    def _create_plus_nfa(self, start: int, end: int) -> Tuple[int, int]:
        """Create plus NFA"""
        star_start, star_end = self._create_kleene_star_nfa(start, end)

        new_start = self._get_next_state_id()
        self.nfa_states[new_start] = NFAState(new_start)
        self.nfa_states[new_start].epsilon_transitions.add(start)

        self.nfa_states[end].epsilon_transitions.add(star_start)

        return new_start, star_end

    def _create_optional_nfa(self, start: int, end: int) -> Tuple[int, int]:
        """Create an optional NFA"""
        new_start = self._get_next_state_id()
        self.nfa_states[new_start] = NFAState(new_start)

        self.nfa_states[new_start].epsilon_transitions.add(start)
        self.nfa_states[new_start].epsilon_transitions.add(end)

        return new_start, end

    def _epsilon_closure(self, states: Set[int]) -> Set[int]:
        """Calculate the epsilon closure"""
        closure = set(states)
        stack = list(states)

        while stack:
            state = stack.pop()
            if state in self.nfa_states:
                for next_state in self.nfa_states[state].epsilon_transitions:
                    if next_state not in closure:
                        closure.add(next_state)
                        stack.append(next_state)

        return closure

    def _nfa_to_dfa(self, nfa_start: int):
        """Convert NFA to DFA"""
        initial_closure = self._epsilon_closure({nfa_start})
        initial_is_final = any(self.nfa_states[s].is_final for s in initial_closure if s in self.nfa_states)

        dfa_start_id = 0
        self.dfa_states[dfa_start_id] = DFAState(dfa_start_id, initial_closure, initial_is_final)

        state_mapping = {frozenset(initial_closure): dfa_start_id}
        unprocessed = [initial_closure]
        next_dfa_id = 1

        while unprocessed:
            current_nfa_states = unprocessed.pop(0)
            current_dfa_id = state_mapping[frozenset(current_nfa_states)]

            for symbol in self.alphabet:
                next_nfa_states = set()

                for nfa_state in current_nfa_states:
                    if nfa_state in self.nfa_states:
                        # 使用get方法安全获取转换
                        next_nfa_states.update(self.nfa_states[nfa_state].transitions.get(symbol, set()))

                if next_nfa_states:
                    next_closure = self._epsilon_closure(next_nfa_states)
                    next_closure_frozen = frozenset(next_closure)

                    if next_closure_frozen not in state_mapping:
                        is_final = any(self.nfa_states[s].is_final for s in next_closure if s in self.nfa_states)
                        state_mapping[next_closure_frozen] = next_dfa_id
                        self.dfa_states[next_dfa_id] = DFAState(next_dfa_id, next_closure, is_final)
                        unprocessed.append(next_closure)
                        next_dfa_id += 1

                    target_dfa_id = state_mapping[next_closure_frozen]
                    self.dfa_states[current_dfa_id].transitions[symbol] = target_dfa_id

    def _build_dfa(self):
        """Build DFA"""
        print(f"is working on regular expressions '{self.regex_pattern}' Build DFA...")

        processed_pattern = self._preprocess_regex(self.regex_pattern)

        nfa_start, _ = self._build_nfa_from_regex(processed_pattern)

        self._nfa_to_dfa(nfa_start)

        print(f"The DFA construction is completed！")
        print(f"The number of NFA states: {len(self.nfa_states)}")
        print(f"Number of DFA states: {len(self.dfa_states)}")
        print(f"Alphabet: {self.alphabet}")

    def _preprocess_regex(self, pattern: str) -> str:
        """Preprocess regular expressions"""
        # 扩展元字符
        processed = pattern
        processed = processed.replace('\\d', '[0-9]')
        processed = processed.replace('\\w', '[a-zA-Z0-9_]')
        processed = processed.replace('\\s', '[ \\t\\n\\r\\f\\v]')

        # 处理字符类
        processed = self._expand_char_classes(processed)

        return processed

    def _expand_char_classes(self, pattern: str) -> str:
        """扩展字符类表示法"""
        result = []
        i = 0
        while i < len(pattern):
            if pattern[i] == '[':
                # 找到匹配的右括号
                j = i + 1
                while j < len(pattern) and pattern[j] != ']':
                    j += 1
                if j < len(pattern):
                    char_class = pattern[i + 1:j]
                    # 简单处理字符类，这里只处理了连字符情况
                    expanded = self._expand_char_range(char_class)
                    result.append(f"(?:{expanded})")
                    i = j + 1
                    continue
            result.append(pattern[i])
            i += 1
        return ''.join(result)

    def _expand_char_range(self, char_class: str) -> str:
        """扩展字符类中的范围表示"""
        result = []
        i = 0
        while i < len(char_class):
            if i + 2 < len(char_class) and char_class[i + 1] == '-':
                start = char_class[i]
                end = char_class[i + 2]
                # 生成范围内的所有字符
                for c in range(ord(start), ord(end) + 1):
                    result.append(chr(c))
                i += 3
            else:
                result.append(char_class[i])
                i += 1
        # 将字符类转换为或表达式
        return '|'.join([re.escape(c) for c in result])

    def match(self, input_string: str) -> bool:
        """Check if the string matches the regular expression (DFA version)"""
        current_state = 0  # 初始状态

        for char in input_string:
            if char not in self.alphabet:
                return False

            # 检查当前状态是否有对应字符的转换
            if current_state in self.dfa_states:
                transitions = self.dfa_states[current_state].transitions
                if char in transitions:
                    current_state = transitions[char]
                else:
                    return False  # 没有匹配的转换
            else:
                return False  # 无效状态

        # 检查最终状态是否为接受状态
        return (current_state in self.dfa_states and
                self.dfa_states[current_state].is_final)

    def get_dfa_info(self) -> Dict:
        """Get DFA information"""
        transitions_info = {}
        for state_id, state in self.dfa_states.items():
            transitions_info[state_id] = {
                'is_final': state.is_final,
                'transitions': dict(state.transitions),
                'nfa_states': list(state.nfa_states)
            }

        return {
            'regex_pattern': self.regex_pattern,
            'alphabet': list(self.alphabet),
            'states_count': len(self.dfa_states),
            'start_state': 0,
            'states': transitions_info
        }



# test
if __name__ == "__main__":
    dfa = RegexDFA("a*b+")

    test_strings = ["b", "ab", "aab", "aaabbb", "abc"]
    for test in test_strings:
        result = dfa.match(test)
        status = "✅ match" if result else "❌ Not match"
        print(f"  🧪 '{test}': {status}")