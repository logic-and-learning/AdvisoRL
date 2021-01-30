import matplotlib.pyplot as plt
import pylab
import matplotlib 
import xlrd

loc = "~/Desktop/learning-automata-for-rl-longer/data/office_t1.xlsx"

wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(4)

steps = list()
p25 = list()
p50 = list()
p75 = list()
p25_q = list()
p50_q = list()
p75_q = list()
p25_hrl = list()
p50_hrl = list()
p75_hrl = list()


for row in sheet._cell_values:
    steps.append(row[0])
    p25.append(row[1])
    p50.append(row[2])
    p75.append(row[3])
    p25_q.append(row[4])
    p50_q.append(row[5])
    p75_q.append(row[6])
    p25_hrl.append(row[7])
    p50_hrl.append(row[8])
    p75_hrl.append(row[9])

#p25 = [0,0,0.125,0.13,0.15,0.15,0.2,0.3,0.3,0.5,1]
#p50 = [0,0,0.25,0.3,0.25,0.3,0.4,0.6,0.5,0.9,1]
#p75 = [0,0,0.35,0.5,0.6,0.75,0.75,1,1,1,1]

#steps = [0,100000,200000,300000,400000,500000,600000,700000,800000,900000,1000000]

fig, ax = plt.subplots()
fig.set_figheight(6)
fig.set_figwidth(8)


ax.plot(steps,p25,alpha=0)
ax.plot(steps,p50, color='black', label='JIRP')
ax.plot(steps,p75,alpha=0)
ax.plot(steps,p25_q,alpha=0)
ax.plot(steps,p50_q, color='red', label='Q-learning')
ax.plot(steps,p75_q,alpha=0)
ax.plot(steps,p25_hrl,alpha=0)
ax.plot(steps,p50_hrl, color='blue', label='HRL')
ax.plot(steps,p75_hrl,alpha=0)


ax.grid()

plt.fill_between(steps,p50,p25,color='black',alpha=0.25)
plt.fill_between(steps,p50,p75,color='black',alpha=0.25)
plt.fill_between(steps,p50_q,p25_q,color='red',alpha=0.25)
plt.fill_between(steps,p50_q,p75_q,color='red',alpha=0.25)
plt.fill_between(steps,p50_hrl,p25_hrl,color='blue',alpha=0.25)
plt.fill_between(steps,p50_hrl,p75_hrl,color='blue',alpha=0.25)

ax.set_xlabel('number of training steps', fontsize=22)
ax.set_ylabel('reward', fontsize=22)
plt.ylim(-0.1,1.1)
plt.xlim(0,200000)

plt.locator_params(axis='x',nbins=5)

plt.yticks([0,0.2,0.4,0.6,0.8,1])

plt.gcf().subplots_adjust(bottom=0.15)
plt.gca().legend(('','JIRP','','','Q-learning','','','HRL',''))
plt.legend(loc='upper right',bbox_to_anchor=(1,0.9),prop={'size':22})

ax.tick_params(axis='both', which = 'major', labelsize = 22)

plt.savefig('figure_1.png', dpi=600)
plt.show()

