raw = pydea.io.read_raw(raw_file)
events = pydea.event.extract_events(raw, event_parameters)
# events is array of Event obj

raw.plot_heatmap()
raw.plot_agg_history()
raw.plot_agg_profile()

event.plot_heatmap()
event.plot_agg_history()
event.plot_agg_history()

