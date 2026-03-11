import sys, os, traceback
sys.path.insert(0, r'c:\Users\welcome\Downloads\Gojan AI\Gojan-AI-Voice-Assistant')

try:
    from phase_b_local.services.llm import load_model, generate_answer
    with open('test_result.txt', 'w', encoding='utf-8') as f:
        f.write('Loading Zephyr LLM...\n')
    
    llm = load_model()

    mock_context = '''
    Gojan School of Business and Technology is located at 80 Feet Road, Edapalayam, Redhills, Chennai - 600 052.
    Dharmalingam is our Chairman. Our phone number is +91 7010723984.
    Courses offered include B.E. Computer Science, Mechanical Engineering, and AI-ML.
    '''

    questions = [
        ('Where is Gojan college located?', 'english', mock_context),
        ('Gojan la enna courses iruku?', 'tanglish', mock_context),
        ('What is the library timing?', 'english', mock_context),
    ]

    with open('test_result.txt', 'a', encoding='utf-8') as f:
        for q, lang, ctx in questions:
            f.write(f'Q: {q}\n')
            ans = generate_answer(llm, q, ctx, lang, '')
            f.write(f'A: {ans}\n')
            f.write('-' * 40 + '\n')
            f.flush()

except Exception as e:
    with open('test_result.txt', 'a', encoding='utf-8') as f:
        f.write("\nERROR CRASHED:\n")
        f.write(traceback.format_exc())
