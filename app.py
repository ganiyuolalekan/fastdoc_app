from app_utils import app_meta, divider, st
from app_utils import init_project, return_project_value, delete_project, json_to_dict, dict_to_json, write_out_report

app_meta()

with st.sidebar:
    st.write("FastDoc Document Generation")
    start_project = st.checkbox(
        label="Start Application",
        help="Starts The Demo Application"
    )
    divider()

if start_project:
    display_generated = display_comment = display_delete = False

    with st.sidebar:
        st.markdown("### 1. Input to generate text")
        epic_key = st.text_input(label="Enter Issue key - eg FD-5", value="FD-5")
        project_id = st.number_input(label="Enter any number to store your result with", value=12345),
        doc_type = st.text_input(label="Enter preferred document type", value="Non-technical Document")
        audience = st.text_input(label="Who are your audience", value="Product managers and technical writers")
        goal = st.text_input(label="What's the goal of this document?", value="Develop a non-technical documentation for the product release")
        tone = st.text_input(label="What tone should your document have", value="Professional Write-up")

        submit = st.button(label="Submit")
        divider()

    if submit:
        st.markdown("## Here's the extracted report from Jira".upper())
        divider()
        st.write(write_out_report(epic_key)[0])
        divider()
        st.write("Generating write up...")
        generate_text_res = init_project(dict_to_json({
            'goal': goal,
            'tone': tone,
            'key': epic_key,
            'doc_type': doc_type,
            'audience': audience,
            'project_id': project_id[0],
        }))
        display_generated = True

    if display_generated:
        divider()
        st.markdown("## Here's the generated text".upper())
        divider()
        text = json_to_dict(generate_text_res)['generated_text']
        st.write(text)
        divider()

    with st.sidebar:
        st.markdown("### 2. Comment to improve text".upper())
        comment = st.text_input(label="What would you like to change?", value="Could you make the tone more like a newsletter")
        submit_2 = st.button(label="Submit", key=11)
        divider()

    if submit_2:
        st.write("Generating write up...")
        suggest_text_res = return_project_value(dict_to_json({
            'project_id': project_id[0],
            'user_query': comment
        }))
        display_comment = True

    if display_comment:
        divider()
        st.write(json_to_dict(suggest_text_res)['re-generated_text'])
        divider()

    with st.sidebar:
        st.markdown("### 3. Delete from database".upper())
        delete = st.button(label="Delete", key=22)

    if delete:
        input_delete = delete_project(dict_to_json({
            'project_id': project_id[0]
        }))
        display_delete = True

    if display_delete:
        st.write(delete_project(input_delete))
else:
    with open('README.md', 'r') as f:
        demo_report = f.read()
    st.markdown(demo_report)