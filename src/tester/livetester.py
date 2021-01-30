import os, copy
from datetime import datetime
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

class LiveTester:
    """
        Plotter for one experiment.
    """
    def __init__(self, curriculum, *,
                 label=None, filebasename=None,
                 show=True, keep_open=True,
    ):
        # super(LiveTester, self).__init__()

        self.curriculum = curriculum

        self.label = label
        self.filebasename = filebasename
        self.show = show
        self.keep_open = keep_open

        self.episode_steps   = np.array([], dtype=int)
        self.episode_rewards = np.array([], dtype=float)
        self.episode_perfs   = np.array([], dtype=float)

        # self.sample_steps    = np.array([], dtype=int)
        # self.sample_positive = np.array([], dtype=int)
        # self.sample_negative = np.array([], dtype=int)
        default_traces = {
        'steps':    np.array([], dtype=int),
        'positive': np.array([], dtype=int),
        'negative': np.array([], dtype=int),
        }
        self.traces = {
            var: copy.deepcopy(default_traces)
            for var in [
                'all_traces',
                'new_traces',
                'traces_numerical',
            ]
        }

        self.default_bool_data = {
            'steps': [],
            'values': [],
            'color': None,
        }
        self.bool_datas = {
            var: copy.deepcopy(self.default_bool_data)
            for var in [] # filled when adding vars
        }

        self.current_step = 0

    def start(self):
        if not self.show: return self

        # self.fig, self.axes = plt.subplots()
        # self.fig, self.axes = plt.subplots(3,1, sharex=True)
        self.fig, self.axes = plt.subplots(3,1, sharex=True, gridspec_kw={'height_ratios': [2, 2, 1]})
        self.axes = np.array(self.axes).flatten()

        if self.label is not None:
            self.fig.canvas.set_window_title(self.label)
        self.fig.canvas.get_default_filename = lambda : 'livefigure_{}.{}'.format(
            self.filebasename or 1,
            self.fig.canvas.get_default_filetype(),
        )
        self.old_savefig_directory = mpl.rcParams["savefig.directory"]
        mpl.rcParams["savefig.directory"] = os.path.abspath("../plotdata/")
        self.fig.set_figheight(8)

        self.axes_handles = [[] for ax in self.axes]

        self.axes[0].set_xlim(0, self.curriculum.total_steps)

        self.plottmp = []


        a = 0 #
        self.axes[a].grid()
        self.axes[a].set(xlabel='number of steps', ylabel='reward')
        self.axes[a].set_ylim(-0.1, 1.1)

        self.plot_perfs, = self.axes[a].plot(self.episode_steps, self.episode_perfs,
            color='black',
            drawstyle='steps-post',
            label='performance',
        )
        self.axes_handles[a].append(self.plot_perfs)


        self.traces_plotparam = [
            ('all_traces', 1),
            # ('new_traces', 1),
            # ('traces_numerical', 1),
        ]
        for _,a in self.traces_plotparam: # traces count
            self.axes[a].grid()
            self.axes[a].set(xlabel='number of steps', ylabel='count')

        self.plot_pos = {}
        self.plot_tot = {}
        for var,a in self.traces_plotparam:
            self.plot_pos[var], = self.axes[a].plot(0,0,
                color='#00aa00',
                drawstyle='steps-post',
                label=var+'.positive',
            )
            self.plot_tot[var], = self.axes[a].plot(0,0,
                color='black',
                drawstyle='steps-post',
                label=var+'.total',
            )
            self.axes_handles[a].append(self.plot_tot[var])
            self.axes_handles[a].append(self.plot_pos[var])


        a = 2 # bool variables
        self.axes[a].set(xlabel='number of steps')
        self.axes[a].get_yaxis().set_visible(False)


        self.markers = {}
        self.markers['rm_update'] = mpl.lines.Line2D([], [],
            label='RM update', color='#cc66aa', alpha=0.5,
            marker='|', linestyle='None', markersize=10, markeredgewidth=1.5,
        )
        self.markers['rm_learn'] = mpl.lines.Line2D([], [],
            label='RM relearn', color='#ff6666', alpha=0.5,
            marker='|', linestyle='None', markersize=10, markeredgewidth=1.5,
        )
        # self.markers['rm_learn_failed'] = mpl.lines.Line2D([], [],
        #     label='RM relearn attempt', color='#0000ff', alpha=0.2,
        #     marker='|', linestyle='None', markersize=10, markeredgewidth=1.5,
        # )
        self.markers['rm_refresh'] = mpl.lines.Line2D([], [],
            label='RM refresh', color='#00dd99', alpha=0.2,
            marker='|', linestyle='None', markersize=10, markeredgewidth=1.5,
        )
        for handles in self.axes_handles:
            for marker in self.markers.values():
                handles.append(marker)
            break # just the first one

        for a,handles in enumerate(self.axes_handles):
            self.axes[a].legend(handles=handles)

        plt.ion()

        # self.fig.canvas.draw_idle()
        self.last_update = datetime.now()
        self.last_update_duration = 0
        self.__update(force=True)
        plt.pause(0.0001)

        return self

    def __update(self, force=False):
        if not self.show: return

        start_update_time = datetime.now()
        if not force:
            elapsed_time = (start_update_time-self.last_update).total_seconds()
            max_elapsed_time = max(0.1, min(2, self.last_update_duration*5)) # take only 20% of compute time
            if elapsed_time < max_elapsed_time: return

        for elem in self.plottmp:
            elem.remove()
        self.plottmp.clear()

        for ax in self.axes:
            self.plottmp.append(ax.axvspan(self.current_step, self.curriculum.total_steps,
                facecolor="black", alpha=0.1, zorder=-128,
            ))

        if len(self.episode_steps):
            steps = np.append(self.episode_steps, self.current_step)
            perfs = np.append(self.episode_perfs, self.episode_perfs[-1])
            self.plot_perfs.set_data(steps, perfs)

        for var,a in self.traces_plotparam:
            if len(self.traces[var]['steps']):
                total = self.traces[var]['positive'] + self.traces[var]['negative']

                steps = np.append(self.traces[var]['steps'], self.current_step)
                pos = np.append(self.traces[var]['positive'], self.traces[var]['positive'][-1])
                tot = np.append(total, total[-1])

                self.plot_pos[var].set_data(steps, pos)
                self.plot_tot[var].set_data(steps, tot)
                self.plottmp.append(self.axes[a].fill_between(steps, 0, pos,
                    facecolor=self.plot_pos[var].get_color(),
                    alpha=0.5,
                    step="post",
                ))
                self.plottmp.append(self.axes[a].fill_between(steps, pos, tot,
                    facecolor="#dd0000",
                    alpha=0.5,
                    step="post",
                ))

        a=2
        for v,(var,bool_data) in enumerate(self.bool_datas.items()):
            if not len(bool_data['steps']): continue
            steps = np.append(bool_data['steps'], self.current_step)
            values = np.append(bool_data['values'], bool_data['values'][-1])
            args = {}
            if bool_data['color'] is not None: args['facecolor'] = bool_data['color']
            fill = self.axes[a].fill_between(steps, v, v+values,
                alpha=0.5, step="post",
                label=var,
                **args,
            )
            if bool_data['color'] is None: # not already set
                bool_data['color'] = fill.get_facecolor()
                self.axes_handles[a].insert(0, fill)
            self.plottmp.append(fill)
        self.axes[a].set_ylim(0, len(self.bool_datas))

        for a,handles in enumerate(self.axes_handles):
            self.axes[a].legend(handles=handles)

        self.fig.canvas.draw_idle()
        self.fig.canvas.start_event_loop(0.0001)

        self.last_update = datetime.now()
        self.last_update_duration = (self.last_update-start_update_time).total_seconds()

    def save(filename=None):
        if not self.show: raise RuntimeError("Graph not active.")
        if filename is None:
            filename = os.path.join(mpl.rcParams["savefig.directory"], self.fig.canvas.get_default_filename())
        self.fig.savefig(filename)

    def close(self):
        self.current_step = self.curriculum.total_steps
        if not self.show: return

        plt.ioff()
        if self.keep_open:
            self.__update(True)
            plt.show()
        else:
            self.fig.close()

        mpl.rcParams["savefig.directory"] = self.old_savefig_directory

    def add_reward(self, step, reward, *, force_update=False):
        self.current_step = step
        self.episode_steps   = np.append(self.episode_steps, step)
        self.episode_rewards = np.append(self.episode_rewards, reward)
        perf = np.average(self.episode_rewards[-10:])
        self.episode_perfs   = np.append(self.episode_perfs, perf)
        self.__update(force_update)

    def add_event(self, step, event='rm_update', *, force_update=False):
        self.current_step = step
        if self.show and event in self.markers.keys():
            marker = self.markers[event]
            for ax in self.axes[:-1]:
                ax.axvline(x=step, color=marker.get_color(), alpha=marker.get_alpha(), zorder=-32)
        self.__update(force_update)

    def add_traces_size(self, step, traces, var='all_traces', *, force_update=False):
        self.current_step = step
        if not var in self.traces.keys(): return
        traces_data = self.traces[var]
        traces_data['steps']    = np.append(traces_data['steps'], step)
        traces_data['positive'] = np.append(traces_data['positive'], len(traces.positive))
        traces_data['negative'] = np.append(traces_data['negative'], len(traces.negative))
        self.__update(force_update)

    def add_bool(self, step, var, val, *, force_update=False):
        self.current_step = step
        if not var in self.bool_datas.keys():
            self.bool_datas[var] = copy.deepcopy(self.default_bool_data)
        self.bool_datas[var]['steps'].append(step)
        self.bool_datas[var]['values'].append(bool(val))
        self.__update(force_update)

    def __enter__(self):
        self.start()
        return self
    def __exit__(self, type, value, traceback):
        self.close()
