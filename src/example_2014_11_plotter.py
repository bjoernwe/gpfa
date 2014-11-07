import plotter


def main():
    plotter.plot(plotter._example_func, x=0, y=range(10), repetitions=10)


if __name__ == '__main__':
    main()
