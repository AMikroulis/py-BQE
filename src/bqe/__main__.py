import argparse

from bqe import MODELS, UNIT_PRIORS, TARGET_ACCEPT, main, simulate


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', nargs='?', default='sample.txt')
    parser.add_argument('--warmup', type=int, default=500)
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--model', default='dep', choices=list(MODELS),
                        help='dep = depression only, daf = with facilitation')
    parser.add_argument('--n-max', type=int, default=10,
                        help='marginalise the number of release sites over 1...n-max')
    parser.add_argument('--units', default='mV', choices=list(UNIT_PRIORS),
                        help='amplitude unit system (sets default prior bounds)')
    parser.add_argument('--a-range', type=float, nargs=2, metavar=('LO', 'HI'),
                        help='override prior range for quantal amplitude a')
    parser.add_argument('--siga-range', type=float, nargs=2, metavar=('LO', 'HI'),
                        help='override prior range for quantal SD siga')
    parser.add_argument('--sigb-range', type=float, nargs=2, metavar=('LO', 'HI'),
                        help='override prior range for baseline noise sigb')
    parser.add_argument('--tauf-range', type=float, nargs=2, metavar=('LO', 'HI'),
                        help='override prior range for facilitation tauf [in ms]')
    parser.add_argument('--target-accept', type=float,
                        help=f'NUTS target acceptance, default = {TARGET_ACCEPT} '
                             'Raise it if the summary reports many divergences.')

    sim = parser.add_argument_group('synthetic data (generates and exits)')
    sim.add_argument('--simulate', metavar='OUT.txt',
                     help='write a simulated train to OUT.txt and quit')

    sim.add_argument('--sim-n', type=int, default=5)
    sim.add_argument('--sim-p', type=float, default=0.2)
    sim.add_argument('--sim-f', type=float, default=0.5)
    sim.add_argument('--sim-tauf', type=float, default=100.0)
    sim.add_argument('--sim-tD', type=float, default=300.0)
    sim.add_argument('--sim-a', type=float, default=0.2)
    sim.add_argument('--sim-siga', type=float, default=0.05)
    sim.add_argument('--sim-sigb', type=float, default=0.02)
    sim.add_argument('--sim-pulses', type=int, default=8)
    sim.add_argument('--sim-isi', type=float, default=50.0)
    sim.add_argument('--sim-sweeps', type=int, default=30)
    sim.add_argument('--sim-times', type=float, nargs='+', metavar='MS',
                     help='explicit pulse times [in ms]; overrides --sim-pulses/--sim-isi')
    sim.add_argument('--sim-seed', type=int, default=0)
    args = parser.parse_args()

    if args.simulate:
        simulate(args.simulate,
                n = args.sim_n,
                p = args.sim_p,
                f = args.sim_f,
                tauf = args.sim_tauf,
                tD = args.sim_tD,
                a = args.sim_a,
                siga = args.sim_siga,
                sigb = args.sim_sigb,
                n_pulses = args.sim_pulses,
                isi = args.sim_isi,
                n_sweeps = args.sim_sweeps,
                ts = args.sim_times,
                seed = args.sim_seed)
    else:
        main(args.filename,
                args.warmup,
                args.samples,
                args.chains,
                units = args.units,
                a_range = tuple(args.a_range) if args.a_range else None,
                siga_range = tuple(args.siga_range) if args.siga_range else None,
                sigb_range = tuple(args.sigb_range) if args.sigb_range else None,
                model = args.model,
                n_max = args.n_max,
                tauf_range = tuple(args.tauf_range) if args.tauf_range else None,
                target_accept = args.target_accept)


if __name__ == '__main__':
    cli()
