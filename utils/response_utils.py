
def get_prompt_response(res):
    if getattr(res, "content", None):
        return res.content
    else:
        return res