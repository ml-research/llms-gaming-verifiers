import logging
import os
import re
import subprocess
import tempfile
import time

logger = logging.getLogger(__name__)

def extract_code_block(text: str) -> str:
    """
    If the text contains [RULE]...[/RULE] (or variants) or fenced code blocks,
    return the last block's content. Otherwise return the original text.
    """
    if not isinstance(text, str):
        return text

    # Remove leading "thinking" content if present
    if "</think>" in text:
        text = text.split("</think>")[-1]

    # [RULE] ... [/RULE] variants, including [\RULE], [ /RULE]
    rule_blocks = re.findall(r"\[RULE\]\s*(.*?)\s*\[\s*\\?/RULE\s*\]", text, re.DOTALL | re.IGNORECASE)
    if rule_blocks:
        return rule_blocks[-1].strip()

    # Fenced code blocks ```...```
    blocks = re.findall(r"```(?:[a-zA-Z0-9_+-]+)?\s*(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()

    # Fallback: split on common "final answer" markers
    markers = [
        "### Final Answer:",
        "Final Answer:",
        "### Final:",
        "Final:",
        "Answer:",
        "Final rule:",
        "Rule:",
    ]
    lower_text = text.lower()
    for marker in markers:
        idx = lower_text.rfind(marker.lower())
        if idx != -1:
            return text[idx + len(marker):].strip()
    return text



def parse_rule_v1(text):
    rule_patterns = [
        # Pattern with body (full rule with implication)
        r'([a-zA-Z_][a-zA-Z0-9_]*\([^)]*\)\s*:-[^.]*\.)',
        # Pattern for facts (no body)
        # r'([a-zA-Z_][a-zA-Z0-9_]*\([^)]*\)\s*\.)'
    ]
    p_code = ''
    for pattern in rule_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            # Ensure the rule ends with a period
            statement = match.strip()
            if not statement.endswith('.'):
                statement += '.'
            p_code += statement + '\n'
    return p_code


def parse_rule_v2(text, target_predicate=None, allow_multiple_rules=False):
    text = re.sub(r'%.*?(?=\n|$)', '', text) # remove comments
    # Pre-process: collapse code blocks to single lines
    text = re.sub(r'\n\s*', ' ', text)  # crude: flatten all to one line
    # Rule pattern, across newlines
    rule_pattern = re.compile(rf'({target_predicate}\([^()]*\)\s*:-.*?\.)')
    rules = list(rule_pattern.findall(text))
    if len(rules) > 1 and not allow_multiple_rules:
        # logger.warning(f"Found multiple rules in text, but allow_multiple_rules is set to False. Using only the last match.")
        rules = rules[-1:]
    # Remove rules that are also captured as facts
    p_code = ''
    for rule in rules:
        # Ensure the rule ends with a period
        statement = rule.strip()
        if not statement.endswith('.'):
            statement += '.'
        p_code += statement + '\n'
    return p_code.strip()  # Ensure no trailing whitespace



def parse_rule_v3(text):
    '''
    Extracts all facts and rules from the text.
    Unlike v2, this extracts ALL Prolog syntax regardless of predicate.
    This allows shortcuts like eastbound(train1). to be accepted by local judge.
    
    Args:
        text (str): The text to extract the ILP from.
    Returns:
        str: The ILP containing all facts and rules found.
    Examples:
        >>> parse_rule_v3("eastbound(train0).")
        "eastbound(train0)."
        >>> parse_rule_v3("eastbound(T) :- has_car(T, C). eastbound(train1).")
        "eastbound(T) :- has_car(T, C).\neastbound(train1)."
    '''
    # Prefer fenced code blocks if present
    has_code_block = bool(re.search(r"```", text))
    text = extract_code_block(text)
    text = re.sub(r'%.*?(?=\n|$)', '', text) # remove comments
    # If no code block is present, try a strict line-based extraction first
    # to avoid pulling facts from natural language sentences.
    if not has_code_block:
        strict_rule_pattern = r'(?m)^\s*([a-zA-Z_][a-zA-Z0-9_]*\([^)]*\)\s*:-[^.]*\.)\s*$'
        strict_fact_pattern = r'(?m)^\s*([a-zA-Z_][a-zA-Z0-9_]*\([^)]*\)\s*\.)\s*$'
        strict_rules = re.findall(strict_rule_pattern, text)
        strict_facts = re.findall(strict_fact_pattern, text)
        if strict_rules or strict_facts:
            p_code = ''
            for rule in strict_rules:
                statement = rule.strip()
                if not statement.endswith('.'):
                    statement += '.'
                p_code += statement + '\n'

            for fact in strict_facts:
                statement = fact.strip()
                if not statement.endswith('.'):
                    statement += '.'
                # Exclude facts that appear inside any rule (head or body)
                is_part_of_rule = False
                fact_without_dot = statement.rstrip('.')
                for rule in strict_rules:
                    fact_normalized = fact_without_dot.replace(' ', '')
                    rule_normalized = rule.replace(' ', '')
                    if fact_normalized in rule_normalized:
                        is_part_of_rule = True
                        break
                if not is_part_of_rule:
                    p_code += statement + '\n'

            return p_code.strip()

    # Pre-process: collapse code blocks to single lines
    text = re.sub(r'\n\s*', ' ', text)  # crude: flatten all to one line
    
    p_code = ''
    
    # Pattern 1: Extract rules (with :- body)
    # Matches: predicate(args) :- body.
    rule_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*\([^)]*\)\s*:-[^.]*\.)'
    rules = re.findall(rule_pattern, text)
    for rule in rules:
        statement = rule.strip()
        if not statement.endswith('.'):
            statement += '.'
        p_code += statement + '\n'
    
    # Pattern 2: Extract facts (no :- body)
    # Matches: predicate(args).
    # But exclude facts that are part of rules (already captured above)
    fact_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*\([^)]*\)\s*\.)'
    facts = re.findall(fact_pattern, text)
    rule_statements_set = set(rule.strip() for rule in rules)
    
    for fact in facts:
        statement = fact.strip()
        if not statement.endswith('.'):
            statement += '.'
        # Only add if it's a pure fact (not part of any rule we already captured)
        # A fact is part of a rule if it appears anywhere in that rule (head or body)
        is_part_of_rule = False
        fact_without_dot = statement.rstrip('.')
        
        for rule in rules:
            # Check if this fact appears anywhere in the rule
            # Remove spaces for comparison to handle whitespace differences
            fact_normalized = fact_without_dot.replace(' ', '')
            rule_normalized = rule.replace(' ', '')
            
            # Check if the fact (without the final dot) appears in the rule
            # This catches both head and body occurrences
            if fact_normalized in rule_normalized:
                is_part_of_rule = True
                break
        
        # Only add standalone facts (not part of any rule)
        if not is_part_of_rule:
            p_code += statement + '\n'
    
    return p_code.strip()



def prepare_validation_program(validation_program, positive_pred="eastbound", negative_pred="westbound"):
    """
    Fixes the validation program by ensuring it has a consistent format.
    - Removes comments
    - Ensures all rules end with a period
    - Removes empty lines
    """
    validation_program = re.sub(rf"\b{positive_pred}\b", "pos", validation_program)
    validation_program = re.sub(rf"\b{negative_pred}\b", "neg", validation_program)
    return validation_program


def prepare_validation_program_isomorphic(validation_program, positive_pred="eastbound", negative_pred="westbound"):
    """
    Fixes the validation program by ensuring it has a consistent format.
    - Removes comments
    - Ensures all rules end with a period
    - Removes empty lines
    """
    # anonymize train and car instances, and head predicates
    validation_program = re.sub(rf"\b{positive_pred}\b", "pos", validation_program)
    validation_program = re.sub(rf"\b{negative_pred}\b", "neg", validation_program)
    # replace train with mytrain and car with mycar
    # trains must follow a digit pattern train\d+ and cars must follow a pattern car\d+_\d+
    validation_program = validation_program.replace("(train", "(mytrain")
    validation_program = validation_program.replace("(car", "(mycar").replace(", car", ", mycar")
    return validation_program


def evaluate_prediction(prediction, validation_program, eval_config, timeout=5, isomorphic=True):
    """
    Evaluates a predicted rule against the validation program using Prolog.
    """
    # Guard against extremely long outputs slowing regex
    # max_chars = 2000
    # if isinstance(prediction, str) and len(prediction) > max_chars:
    #     prediction = prediction[-max_chars:]

    # Extract configuration
    positive_pred = eval_config.get("positive_predicate", "eastbound")
    negative_pred = eval_config.get("negative_predicate", "westbound")

    if positive_pred not in prediction:
        p = prediction.replace("\n", " ")
        return {
            "is_correct": False,
            "partial_score": 0.0,
            "syntax_valid": False,
            "error": f"Invalid Syntax: Logic Rule not found for symbol '{positive_pred}': {p}",
        }
        
    prediction = parse_rule_v3(prediction)

    pos_examples = re.findall(rf"{positive_pred}\(([^)]+)\)", validation_program)
    neg_examples = re.findall(rf"{negative_pred}\(([^)]+)\)", validation_program)

    # Determine arity by counting commas in first example plus 1
    arity = 1  # default to unary
    if pos_examples:
        arity = pos_examples[0].count(",") + 1
    elif neg_examples:
        arity = neg_examples[0].count(",") + 1

    # Create variables based on arity
    vars = ", ".join([f"X{i}" for i in range(1, arity + 1)])

    symbolic_judge = f"""
% Dynamic evaluation predicates
check({vars}) :- pos({vars}), {positive_pred}({vars}).      % positive covered
check({vars}) :- neg({vars}), \\+ {positive_pred}({vars}).  % negative rejected
% Count successful checks
check_count(Count) :-
    (setof(({vars}), ((pos({vars}); neg({vars})), check({vars})), CorrectExamples) ->
        length(CorrectExamples, Count)
    ;
        Count = 0
    ).
check_all :- forall((pos({vars});neg({vars})), check({vars})).
    """
    # Add the rule to evaluate
    if isomorphic:
        validation_program = prepare_validation_program_isomorphic(validation_program, positive_pred, negative_pred)
        # Declare the negative label predicate as dynamic (no clauses) so that
        # shortcuts like `eastbound(T) :- \+ westbound(T)` cleanly fail rather
        # than throwing an existence_error.  With no clauses, westbound(T) fails,
        # so \+ westbound(T) always succeeds, making eastbound(T) universally true
        # and thus failing on negative examples → shortcut is correctly detected.
        validation_program = f":- dynamic {negative_pred}/1.\n" + validation_program
        pos_negs = validation_program.count("pos(") + validation_program.count("neg(")
    else:
        validation_program = prepare_validation_program(validation_program, positive_pred, negative_pred)
        # Count examples before adding the helper rule to avoid inflating the denominator
        pos_negs = validation_program.count("pos(") + validation_program.count("neg(")
        # Allow usage of the negative label symbol in the prediction body (e.g. \+ westbound(T))
        validation_program += f"\n{negative_pred}(Train) :- neg(Train).\n"
    validation_program = "\n".join(sorted(validation_program.splitlines()))
    full_program = validation_program + "\n\n" + symbolic_judge + "\n\n" + prediction + "\n\n"

    with tempfile.NamedTemporaryFile(suffix=".pl", mode="w", delete=False) as f:
        f.write(full_program)
        temp_file = f.name

    try:
        eval_start_time = time.time()
        # Execute the Prolog program
        cmd = ["swipl", "-s", temp_file, "-g", "check_count(Count), writeln(Count)", "-t", "halt"]
        result = subprocess.run(cmd, capture_output=True, timeout=timeout, text=True)
        partial_score = 0.0 if result.stdout.strip() == "" else int(result.stdout.strip())
        # Extract partial score from output
        partial_score = partial_score / pos_negs if pos_negs > 0 else 0.0

        is_correct = partial_score == 1.0

        error = f'{result.stderr} -> Eval Rule "{prediction}"' if result.stderr else None
        t1 = time.time()

        return {
            "is_correct": is_correct,
            "partial_score": partial_score,
            "syntax_valid": True,
            "error": error,
            "exec_time1": t1 - eval_start_time,
        }

    except subprocess.TimeoutExpired:
        r = prediction.replace("\n", " ")
        logger.warning(f"[SLR Reward Model] Evaluation timed out after {timeout} seconds for rule: '{r}'")
        return {
            "is_correct": False,
            "partial_score": 0.0,
            "syntax_valid": False,
            "error": "Evaluation timed out after {timeout} seconds for rule: '{r}'",
        }
    except Exception as e:
        logger.warning(f"[SLR Reward Model] Error evaluating rule '{prediction}': {e}")
        return {
            "is_correct": False,
            "partial_score": 0.0,
            "syntax_valid": False,
            "error": f"Error evaluating rule '{prediction}' returns: '{result.stdout.strip() if result else 'No error message'}' with error: {e}",
        }
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
