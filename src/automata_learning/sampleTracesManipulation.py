from Traces import Traces

def main():
    tr = Traces(positive=set([(1,2,3,3), (1,1,2)]),
                negative=set([(1,2,3), (0,0,1)])
                )
    tr.export_traces("exported_traces.txt")

if __name__ == '__main__':
    main()