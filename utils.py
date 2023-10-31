from modules.variables import fastdoc_url
from modules.functions import (
    dict_to_json, generate_text, regenerate_report, json_to_dict, write_out_report
)


def init_project(json_input):
    """
    Initializes a new project and creates in id for it in our local database
    """

    try:
        keys = json_to_dict(json_input)

        content, issue_key_type = write_out_report(keys['scope'])

        try:
            temp = keys['temperature']
        except KeyError:
            temp = 'variable'

        try:
            url = keys['url']
        except KeyError:
            url = fastdoc_url

        try:
            org = keys['org']
        except KeyError:
            org = "fastdoc"

        goal = None if keys['goal'] == "" else keys['goal']

        result = generate_text(
            keys['project_id'],
            content,
            keys['tone'],
            keys['doc_type'],
            url,
            org,
            goal,
            temperature=temp
        )

        title = result['title']
        text = result['generated_text']

        return dict_to_json({
            'status': 200,
            'title': title,
            'generated_text': text,
            'issue_key_type': issue_key_type,
            'log': "Successfully generated report!!!"
        })
    except Exception as e:
        return dict_to_json({
            'status': 503,
            'log': f"Program failed with exception {e}"
        })


def return_project_value(json_input):
    """Responsible for continuous query for a particular database"""

    try:
        keys = json_to_dict(json_input)

        re_gen_report = regenerate_report(
            keys['project_id'],
            keys['user_query']
        )

        if type(re_gen_report) == str:
            return dict_to_json({
                're-generated_text': re_gen_report,
                'status': 200,
                'log': "Successfully re-generated report!!!"
            })
        else:
            return re_gen_report
    except Exception as e:
        return dict_to_json({
            'status': 503,
            'log': f"Program failed with exception {e}"
        })


def delete_project(json_input):
    """Responsible for delete a project from ML database"""

    return dict_to_json({
        'status': 200,
        'log': "Successfully deleted projects!!!"
    })
