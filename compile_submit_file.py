import pandas as pd
import argparse


def main(args):
    result_df = pd.read_csv(args.output_scores)

    if args.hflip_output_scores:
        hflip_result_df = pd.read_csv(args.hflip_output_scores)

        base_scores = result_df['output_score']/2 + hflip_result_df['output_score']/2
        parts_scores = result_df['parts_output']/2 + hflip_result_df['parts_output']/2
    else:
        base_scores = result_df['output_score']
        parts_scores = result_df['parts_output']

    test_df = pd.read_csv(args.test_csv_path)
    test_df['liveness'] = 0.25 * base_scores + 0.75 * parts_scores

    THR = 0.7
    st = 0.005
    test_df['split'] = test_df['path'].apply(lambda x: x.split('/')[0])
    val_df = test_df[test_df.split == 'val']
    test_df = test_df[(test_df.split == 'test')]
    val_df.loc[val_df['liveness'] < THR, 'liveness'] = THR-st
    val_df.loc[val_df['liveness'] >= THR, 'liveness'] = THR + st

    submit_df = pd.concat([val_df, test_df])
    submit_df[['path', 'liveness']].to_csv(args.out_path, sep=' ', header=None, index=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--test_csv_path',
                        type=str,
                        default='test_out.csv',
                        help='Path to test csv file with image pathes')
    parser.add_argument('--output_scores',
                        type=str,
                        help='Path to test output results')
    parser.add_argument('--hflip_output_scores',
                        default=None,
                        type=str,
                        help='Path to test output results with horizontal flip')
    parser.add_argument('--out_path',
                        type=str,
                        default='submission.txt',
                        help='Output path with submit format')
    args = parser.parse_args()
    main(args)