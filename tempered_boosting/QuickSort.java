//
// Various flavours of Quicksort

import java.util.Vector;

public class QuickSort {
  private static long comparisons = 0;
  private static long exchanges = 0;

  public static void quicksort(double[] a) {
    shuffle(a);
    quicksort(a, 0, a.length - 1);
  }

  public static void quicksort(double[] a, String[] s) {
    shuffle(a, s);
    quicksort(a, s, 0, a.length - 1);
  }

  public static void quicksort(double[] a, int left, int right) {
    if (right <= left) return;
    int i = partition(a, left, right);
    quicksort(a, left, i - 1);
    quicksort(a, i + 1, right);
  }

  public static void quicksort(double[] a, String[] s, int left, int right) {
    if (right <= left) return;
    int i = partition(a, s, left, right);
    quicksort(a, s, left, i - 1);
    quicksort(a, s, i + 1, right);
  }

  private static int partition(double[] a, int left, int right) {
    int i = left - 1;
    int j = right;
    while (true) {
      while (less(a[++i], a[right]))
        ;
      while (less(a[right], a[--j])) if (j == left) break;
      if (i >= j) break;
      exch(a, i, j);
    }
    exch(a, i, right);
    return i;
  }

  private static int partition(double[] a, String[] s, int left, int right) {
    int i = left - 1;
    int j = right;
    while (true) {
      while (less(a[++i], a[right]))
        ;
      while (less(a[right], a[--j])) if (j == left) break;
      if (i >= j) break;
      exch(a, s, i, j);
    }
    exch(a, s, i, right);
    return i;
  }

  private static boolean less(double x, double y) {
    comparisons++;
    return (x < y);
  }

  private static void exch(double[] a, int i, int j) {
    exchanges++;
    double swap = a[i];
    a[i] = a[j];
    a[j] = swap;
  }

  private static void shuffle(double[] a) {
    int N = a.length;
    for (int i = 0; i < N; i++) {
      int r = i + (int) (Math.random() * (N - i));
      exch(a, i, r);
    }
  }

  private static void exch(double[] a, String[] s, int i, int j) {
    exchanges++;
    double swap = a[i];
    a[i] = a[j];
    a[j] = swap;

    String ssw = s[i];
    s[i] = s[j];
    s[j] = ssw;
  }

  private static void shuffle(double[] a, String[] s) {
    int N = a.length;
    for (int i = 0; i < N; i++) {
      int r = i + (int) (Math.random() * (N - i));
      exch(a, s, i, r);
    }
  }
}
