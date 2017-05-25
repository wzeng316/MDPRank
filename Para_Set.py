import yaml

Para={}
Para['version'] = 'V1'
Para['dataset'] = 'MSLR-WEB10K'
Para['model'] = 'Imitation_10'
Para['Nfeature'] = 136
Para['Learningrate'] = 0.0001
Para['Nepisode'] = 100
Para['Lenepisode'] = 10

Para_file = open('Para_info.yml', 'w+')
yaml.dump(Para, Para_file)

Para = yaml.load(file('Para_info.yml'))
print Para