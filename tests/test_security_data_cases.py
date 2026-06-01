from cognicore.envs.data.code_cases import HARD_CASES
from cognicore.envs.data.real_code_cases import REAL_CODE_CASES


def test_real_code_cases_no_literal_api_key_secret():
    hardcoded_secret_case = next((case for case in REAL_CODE_CASES if case["bug_type"] == "hardcoded_secret"), None)
    assert hardcoded_secret_case is not None
    assert "hardcoded-api-key-do-not-do-this" not in hardcoded_secret_case["code"]
    assert "TOKEN = 'replace-me-with-real-token'" in hardcoded_secret_case["code"]


def test_code_hard_10_avoids_eval_in_buggy_code():
    code_hard_10 = next((case for case in HARD_CASES if case.id == "code_hard_10"), None)
    assert code_hard_10 is not None
    assert "\n    eval(" not in code_hard_10.buggy_code
    assert "ast.literal_eval" in code_hard_10.buggy_code
    assert code_hard_10.fix_type == "error_handling"
