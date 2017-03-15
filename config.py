
gconfig = {
    'tags_newline': [
        'tag',
        'tag_next',
        'tag_prev',
        'size_tag',
        'size_tag_next',
        #'len_until_lexem'
    ],
    'tags_space': [
        'tag',
        'tag_next',
        'tag_prev',
        'size_tag',
        'parent_count',
        #'len_until_lexem'
    ],
    'tags_tab': [
        'tag',
        #'tag_next',
        #'tag_prev',
        #'size_tag',
        'parent_count',
        #'len_until_lexem',
        'count_lexems'
    ],

    'categorial_features': [
        'tag',
        'tag_next',
        'tag_prev',
    ],


    'balance': {
        'train': False,
        'test': False,
    },
    'parent_count': 1,

    #'prep_scale': False,
    #'files2process': 100,
    'print_prediction_for_each': False,
    'print_prediction_for_each_newline': False,
    'print_prediction_for_each_space': False,
    'print_prediction_for_each_tab': False,
    'print_ast': False,
    'debug_mode': False,

}
for i in range(gconfig['parent_count']):
    name = 'parent_{}'.format(i+1)
    for key in ['tags_newline', 'tags_space', 'tags_tab', 'categorial_features']:
        gconfig[key].append(name)

