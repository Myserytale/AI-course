import random


def gen_candidate(weight, cap, prices):
    n = len(weight)
    output_list = [0] * n
    profit = 0

    for i in range(n):
        if weight[i] <= cap:
            take = random.randint(0, 1)
            output_list[i] = take
            if take == 1:
                cap -= weight[i]
                profit += prices[i]

    return output_list, profit


def main():
    weight = [46, 40, 42, 38, 10]
    cap = 80
    prices = [12, 19, 19, 15, 8]

    best_list, best_profit = gen_candidate(weight, cap, prices)

    for _ in range(1000):
        o_list2, profit2 = gen_candidate(weight, cap, prices)
        if profit2 > best_profit:
            best_list, best_profit = o_list2, profit2

    print("best picks:", best_list)
    print("best profit:", best_profit)


if __name__ == "__main__":
    main()