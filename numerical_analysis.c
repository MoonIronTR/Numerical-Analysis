#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

#define EPSILON 0.000001



/* ---------- Parser durumu yapýsý ---------- */
typedef struct {
    const char *expr;   /* current expression pointer */
    double      x;      /* current x value */
} Parser;


/* ---------- Prototipler ---------- */
char peek(Parser *p);
char get(Parser *p);
double number(Parser *p);
double factor(Parser *p);
double parse_function(Parser *p);
double power(Parser *p);
double term(Parser *p);
double expr_parser(Parser *p);
double evaluate(const char *expression, double val);
void preprocess_log_base(const char *input, char *output);
void bisection();
void regula_falsi();
void newton_raphson();

/* ---------- Prototipler (lineer cebir) ---------- */
double **alloc_matrix(int n);
void free_matrix(double **m, int n);
void invert_matrix();
void cholesky_alu();
void forward_substitution ();
void backward_substitution(double **U, double *y, double *x, int n);
void gauss_seidel();


/* ---------- Prototipler (7-10) ---------- */
void numerical_derivative();
void trapezoidal();
void simpson_one_third();
void simpson_three_eighth();
double factorial_int(int n);
void gregory_newton_interpolation();
void menu();



int main() {
    menu();
    return 0;
}


void menu() {
    int choice;
    int show_guide = 1;

    while (1) {
        if (show_guide) {
            printf(" \t\t\t ========== FUNCTION INPUT GUIDE ==========\n");
            printf("General instructions for entering a function:\n");
            printf(" - Use '*' for multiplication. Use parentheses '(' and ')' to group terms.\n");
            printf(" - Supported math functions: sin, cos, tan, asin, acos, atan, log, exp, sqrt\n");
            printf(" - To specify logarithms with a custom base, use: log_base(...), e.g., log_2(x+1)\n");
            printf(" - Constants: pi, e      | Variable: x\n");
            printf(" - Exponentiation is written using '^'. For example: x^2 means x squared.\n\n");
            printf("Example 1: sin(3*x) + log_2(x)\n");
            printf("Example 2: log_10(sqrt(x^2 + 1))\n");
            printf("Example 3: exp(-x^2) + cos(x)\n");
            printf("Example 4: log_5(sin(5*x^2 + sin(5*x)))\n\n");
            printf("\t\t\t ========== IMPORTANT ==========:\n");
            printf(" - When entering functions for root-finding, integration, or differentiation,\n");
            printf("   be sure they are valid over the interval you provide.\n");
            printf(" - Root-finding stops when error < EPSILON (%.5f).\n", EPSILON);
            printf(" - For interpolation, x values must be equally spaced.\n");
            printf(" - Always enter numerical input where prompted (e.g., initial guess, interval, etc.).\n\n");
        }

        printf("\t\t\t === Numerical Methods ===\n");
        printf("1.  Bisection\n");
        printf("2.  Regula Falsi\n");
        printf("3.  Newton-Raphson\n");
        printf("4.  Matrix Inversion\n");
        printf("5.  Cholesky-ALU Method\n");
        printf("6.  Gauss-Seidel\n");
        printf("7.  Numerical Derivative (forward/backward/central)\n");
        printf("8.  Simpson Integration (1/3 & 3/8)\n");
        printf("9.  Trapezoidal Integration\n");
        printf("10. Gregory-Newton Interpolation\n");
        printf("11. Show Input Guide Again\n");
        printf("0.  Exit\n");
        printf("Select [1-11] or 0 to exit: ");

        scanf("%d", &choice);
        show_guide = 0;

        switch (choice) {
            case 0:
                printf("Exiting...\n");
                return;

            case 1:
                system("cls");
                bisection();
                break;

            case 2:
                system("cls");
                regula_falsi();
                break;

            case 3:
                system("cls");
                newton_raphson();
                break;

            case 4:
                system("cls");
                invert_matrix();
                break;

            case 5:
                system("cls");
                cholesky_alu();
                break;

            case 6:
                system("cls");
                gauss_seidel();
                break;

            case 7:
                system("cls");
                numerical_derivative();
                break;

            case 8:
			    system("cls");
			    printf("Choose Simpson type:\n");
			    printf("1 for 1/3 Rule\n");
			    printf("2 for 3/8 Rule\n");
			    int simpson_type;
			    scanf("%d", &simpson_type);
			    if (simpson_type == 1) {
			        simpson_one_third();
			    } else if (simpson_type == 2) {
			        simpson_three_eighth();
			    } else {
			        printf("Invalid Simpson type selected.\n");
			    }
			    break;

            case 9:
                system("cls");
                trapezoidal();
                break;

            case 10:
                system("cls");
                gregory_newton_interpolation();
                break;

            case 11:
                system("cls");
                show_guide = 1;
                break;

            default:
                printf("Invalid choice. Please enter a number between 0–11.\n");
                break;
        }
    }
}

/* =========================================================
 *  Yardýmcý parser fonksiyonlarý
 * ========================================================= */

/*  Boþluklarý atlayýp sýradaki karakteri döndürür                                         */
char peek(Parser *p) {
    while (*p->expr == ' ') p->expr++;
    return *p->expr;
}

/*  Bir karakteri tüketir ve döndürür                                                     */
char get(Parser *p) {
    return *p->expr++;
}

/*  Ondalýklý sayýlarý ayrýþtýrýr                                                         */
double number(Parser *p) {
    double result = 0.0;
    while (isdigit(peek(p)) || peek(p) == '.') {
        char c = get(p);
        if (c == '.') {
            double frac = 0.0, base = 0.1;
            while (isdigit(peek(p))) {
                frac  += base * (get(p) - '0');
                base *= 0.1;
            }
            result += frac;
        } else {
            result = result * 10 + (c - '0');
        }
    }
    return result;
}

/*  Fonksiyon adlarýný (sin, cos, …) yakalar ve hesaplar                                   */
double parse_function(Parser *p) {
    char fname[10] = {0};
    int  i = 0;
    while (isalpha(peek(p))) fname[i++] = get(p);

    if (strcmp(fname, "sin")  == 0) return sin (factor(p));
    if (strcmp(fname, "cos")  == 0) return cos (factor(p));
    if (strcmp(fname, "tan")  == 0) return tan (factor(p));
    if (strcmp(fname, "asin") == 0) return asin(factor(p));
    if (strcmp(fname, "acos") == 0) return acos(factor(p));
    if (strcmp(fname, "atan") == 0) return atan(factor(p));
    if (strcmp(fname, "log")  == 0) return log (factor(p));
    if (strcmp(fname, "exp")  == 0) return exp (factor(p));
    if (strcmp(fname, "sqrt") == 0) return sqrt(factor(p));

    printf("Undefined function: %s\n", fname);
    exit(1);
}

/*  Faktör (sayý, sabit, parantez, fonksiyon, unary -) çözer                               */
double factor(Parser *p) {
    if (peek(p) == '(') {
        get(p);
        double res = expr_parser(p);
        if (peek(p) == ')') get(p);
        return res;
    }
    if (isalpha(peek(p))) {
        if (strncmp(p->expr, "pi", 2) == 0) { p->expr += 2; return M_PI; }
        if (strncmp(p->expr, "e",  1) == 0) { p->expr += 1; return M_E;  }
        if (strncmp(p->expr, "x",  1) == 0) { p->expr += 1; return p->x; }
        return parse_function(p);
    }
    if (isdigit(peek(p)) || peek(p) == '.') return number(p);
    if (peek(p) == '-') { get(p); return -factor(p); }
    printf("Invalid char: %c\n", peek(p));
    exit(1);
}

/*  Üst alma iþlemlerini çözer                                                            */
double power(Parser *p) {
    double base = factor(p);
    while (peek(p) == '^') {
        get(p);
        base = pow(base, factor(p));
    }
    return base;
}

/*  Çarpma / bölme iþlemlerini çözer                                                      */
double term(Parser *p) {
    double res = power(p);
    while (peek(p) == '*' || peek(p) == '/') {
        if (get(p) == '*') res *= power(p);
        else              res /= power(p);
    }
    return res;
}

/*  Toplama / çýkarma seviyesini çözer                                                    */
double expr_parser(Parser *p) {
    double res = term(p);
    while (peek(p) == '+' || peek(p) == '-') {
        if (get(p) == '+') res += term(p);
        else              res -= term(p);
    }
    return res;
}

/*  Ýfadeyi, verilen x deðeri için hesaplar                                     */
double evaluate(const char *expression, double val) {
    Parser p;
    p.expr = expression;
    p.x    = val;
    return expr_parser(&p);
}


/*  log_b(expr)’i  (log(expr)/log(b)) biçimine dönüþtürür                                 */
void preprocess_log_base(const char *input, char *output) {
    const char *p = input;
    char       *q = output;

    /*  Tüm karakterleri tarar                                                            */
    while (*p) {
        /*  log_ kalýbý tespit edilir                                                     */
        if (strncmp(p, "log_", 4) == 0) {
            p += 4;
            char base[20] = {0};
            int  bi = 0;

            /*  Taban rakamlarý okunur                                                    */
            while (*p && *p != '(' && bi < 19) base[bi++] = *p++;
            base[bi] = '\0';

            /*  Ardýndan '(' beklenir                                                     */
            if (*p == '(') {
                int cnt = 1;
                const char *sub = ++p;
                /*  Parantez dengesi yapýlýr                                              */
                while (*p && cnt) {
                    if (*p == '(') cnt++;
                    else if (*p == ')') cnt--;
                    p++;
                }
                size_t len = p - sub - 1;

                /*  (log(sub)/log(base)) formatý yazýlýr                                  */
                q += sprintf(q, "(log(");
                strncpy(q, sub, len); q += len;
                q += sprintf(q, ")/log(%s))", base);
            } else {
                printf("Bad log base format.\n");
                exit(1);
            }
        } else {
            *q++ = *p++;
        }
    }
    *q = '\0';
}

double factorial_int(int n) {
    double f = 1.0;
    int i;
    /*  1’den n’e çarpým                                                                 */
    for (i = 2; i <= n; i++) f *= i;
    return f;
}

/*
   c = (a + b) / 2
   eger f(a)*f(c) < 0 ise b = c, yoksa a = c
   durma kosulu: b - a < EPSILON

*/

void bisection() {
    char raw[256], func[512];
    double a, b;

    getchar();  // Önceki scanf'ten kalan '\n' karakterini temizle

    printf("Enter function f(x):\n");
    fgets(raw, sizeof(raw), stdin);
    raw[strcspn(raw, "\n")] = '\0';        // Satýr sonunu sil
    preprocess_log_base(raw, func);        // log_ tabanlarýný iþle

    printf("Enter interval [a b]:\n");
    printf(" a = ");
    scanf("%lf", &a);
    printf(" b = ");
    scanf("%lf", &b);

    double fa = evaluate(func, a);
    double fb = evaluate(func, b);

    if (fa * fb >= 0) {
        printf("Bisection not possible: f(a) and f(b) must have opposite signs.\n");
        return;
    }

    printf("\n%-6s %-10s %-10s %-10s %-10s %-10s %-10s\n",
           "Iter", "a", "b", "c", "f(a)", "f(b)", "|b-a|");

    int    iter = 0;
    double c    = a;
    double fc   = fa;
    int    done = 0;

    while ((b - a) >= EPSILON && !done) {
        c  = 0.5 * (a + b);
        fc = evaluate(func, c);

        printf("%-6d %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f\n",
               iter, a, b, c, fa, fb, fabs(b - a));

        if (fabs(fc) < EPSILON) {
            done = 1;
        } else {
            if (fa * fc < 0) {
                b  = c;
                fb = fc;
            } else {
                a  = c;
                fa = fc;
            }
            iter++;
        }
    }

    printf("\nRoot approximation: %.6f\n", c);
}

/*
 c = (a*f(b) - b*f(a)) / (f(b) - f(a))
   araligi Ikiye Bolme gibi guncelle
   durma kosulu: b - a < EPSILON
*/

void regula_falsi() {
    char raw[256], func[512];
    double a, b;

    getchar(); // buffer temizliði
    printf("Enter function f(x):\n");
    fgets(raw, sizeof(raw), stdin);
    raw[strcspn(raw, "\n")] = '\0';
    preprocess_log_base(raw, func);

    printf("Enter interval [a b]: ");
    scanf("%lf %lf", &a, &b);

    double fa = evaluate(func, a);
    double fb = evaluate(func, b);

    if (fa * fb >= 0) {
        printf("Regula Falsi not possible: same sign f(a), f(b).\n");
        return;
    }

    printf("\n%-6s %-10s %-10s %-10s %-10s %-10s %-10s\n",
           "Iter", "a", "b", "c", "f(a)", "f(b)", "|b-a|");

    int iter = 0;
    int max_iter = 20;
    double c = a;
    double fc = fa;
    int done = 0;

    while (fabs(b - a) >= EPSILON && !done) {
        if (iter >= max_iter) {
            printf("Max iteration limit reached without convergence.\n");
            return;
        }

        c = (a * fb - b * fa) / (fb - fa);
        fc = evaluate(func, c);

        printf("%-6d %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f\n",
               iter, a, b, c, fa, fb, fabs(b - a));

        if (fabs(fc) < EPSILON) {
            done = 1;
        } else {
            if (fa * fc < 0) {
                b = c;
                fb = fc;
            } else {
                a = c;
                fa = fc;
            }
            iter++;
        }
    }

    printf("Root approx: %.6f\n", c);
}

/*
f'(x0) ? (f(x0 + delta) - f(x0 - delta)) / (2 * delta)
   x1 = x0 - f(x0) / f'(x0)
   durma kosulu: fabs(x1 - x0) < EPSILON
   
   */
   
void newton_raphson() {
    char raw[256], func[512];
    double x0;

    getchar(); // buffer temizliði
    printf("Enter function f(x):\n");
    fgets(raw, sizeof(raw), stdin);
    raw[strcspn(raw, "\n")] = '\0';
    preprocess_log_base(raw, func);

    printf("Enter initial guess x0: ");
    scanf("%lf", &x0);

    printf("\n%-6s %-12s %-12s %-12s %-12s\n",
           "Iter", "x_n", "f(x_n)", "f'(x_n)", "|dx|");

    int iter = 0;
    int max_iter = 20;
    double dx = EPSILON * 2;

    while (dx >= EPSILON) {
        if (iter >= max_iter) {
            printf("Max iteration limit reached without convergence.\n");
            return;
        }

        double fx = evaluate(func, x0);
        double fpx = (evaluate(func, x0 + EPSILON) - evaluate(func, x0 - EPSILON)) / (2 * EPSILON);
		
		
		// 
        if (isnan(fx) || isinf(fx) || isnan(fpx) || isinf(fpx)) {
            printf("Complex or undefined value encountered at iter %d, aborting.\n", iter);
            return;
        }
        
        if (fabs(fpx) < EPSILON) {
            printf("Derivative too small, stop.\n");
            return;
        }

        double x1 = x0 - fx / fpx;
        dx = fabs(x1 - x0);

        printf("%-6d %-12.6f %-12.6f %-12.6f %-12.6f\n",
               iter, x0, fx, fpx, dx);

        x0 = x1;
        iter++;
    }

    printf("Root approx: %.6f\n", x0);
}

double **alloc_matrix(int n) {
    double **m = (double **)malloc(n * sizeof(double *));
    int i;
    for (i = 0; i < n; i++)
        m[i] = (double *)calloc(n, sizeof(double)); // <-- DEÐÝÞTÝ
    return m;
}


/*  Matris belleðini serbest býrakýr                                                       */
void free_matrix(double **m, int n) {
    int i;
    for (i = 0; i < n; i++) free(m[i]);
    free(m);
}


void invert_matrix() {
    int n, i, j, k;
    double pivot, factor;

    printf("Enter matrix size n: ");
    scanf("%d", &n);

    double **A   = alloc_matrix(n);
    double **Inv = alloc_matrix(n);
    double **aug = alloc_matrix(n);

    printf("Enter matrix A elements:\n");
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("Enter element A[%d][%d]: ", i, j);
            scanf("%lf", &A[i][j]);
            aug[i][j] = A[i][j];
            if (i == j)
                Inv[i][j] = 1.0;
            else
                Inv[i][j] = 0.0;
        }
    }

    for (i = 0; i < n; i++) {
        pivot = aug[i][i];
        if (fabs(pivot) < EPSILON) {
            printf("[ERROR] Matrix is singular: pivot at row %d is zero. Cannot compute inverse.\n", i);
            free_matrix(A, n); free_matrix(Inv, n); free_matrix(aug, n);
            return;
        }
        for (j = 0; j < n; j++) {
            aug[i][j] /= pivot;
            Inv[i][j] /= pivot;
        }
        for (k = 0; k < n; k++) {
            if (k != i) {
                factor = aug[k][i];
                for (j = 0; j < n; j++) {
                    aug[k][j] -= factor * aug[i][j];
                    Inv[k][j] -= factor * Inv[i][j];
                }
            }
        }
    }

    printf("Inverse matrix:\n");
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++)
            printf("%10.6f ", Inv[i][j]);
        printf("\n");
    }

    free_matrix(A, n); free_matrix(Inv, n); free_matrix(aug, n);
}


/* ------------- L·y = b  (ileri yerleþtirme) -------------------------- */
void forward_substitution(double **L, double *b, double *y, int n)
{
    int i, j;    double sum;
    for (i = 0; i < n; ++i) {
        sum = 0.0;
        for (j = 0; j < i; ++j) sum += L[i][j] * y[j];
        y[i] = (b[i] - sum) / L[i][i];
    }
}

/* ------------- U·x = y  (geri yerleþtirme,  U diyagonali 1) ---------- */
void backward_substitution(double **U, double *y, double *x, int n)
{
    int i, j;    double sum;
    for (i = n-1; i >= 0; --i) {
        sum = 0.0;
        for (j = i+1; j < n; ++j) sum += U[i][j] * x[j];
        x[i] = y[i] - sum;          /*  Uii = 1  olduðu için /U[i][i] yok */
    }
}

void cholesky_alu() {
    int n;
    int i;
    int j;
    int k;

    double sum;

    // Matris boyutu al
    printf("Enter matrix size n: ");
    scanf("%d", &n);

    // Gerekli matris ve vektörleri oluþtur
    double **A = alloc_matrix(n);
    double **L = alloc_matrix(n);
    double **U = alloc_matrix(n);
    double *c  = (double *)malloc(n * sizeof(double));
    double *y  = (double *)malloc(n * sizeof(double));
    double *x  = (double *)malloc(n * sizeof(double));

    // A matris elemanlarýný oku
    printf("Enter matrix A elements:\n");
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("Enter element A[%d][%d]: ", i, j);
            scanf("%lf", &A[i][j]);
        }
    }

    // Sað taraf vektörünü oku
    printf("\nNow enter the constants vector c:\n");
    for (i = 0; i < n; i++) {
        printf(" You're entering c[%d]: ", i);
        scanf("%lf", &c[i]);
    }
    // L ve U matrislerini sýfýrla
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            L[i][j] = 0.0;
            U[i][j] = 0.0;
        }
    }

    // ALU ayrýþtýrmasý (Crout yöntemi, U köþegen = 1)
    for (i = 0; i < n; i++) {
        for (j = i; j < n; j++) {
            sum = 0.0;
            for (k = 0; k < i; k++) {
                sum = sum + L[j][k] * U[k][i];
            }
            L[j][i] = A[j][i] - sum;
        }

        U[i][i] = 1.0;

        for (j = i + 1; j < n; j++) {
            sum = 0.0;
            for (k = 0; k < i; k++) {
                sum = sum + L[i][k] * U[k][j];
            }
            U[i][j] = (A[i][j] - sum) / L[i][i];
        }
    }

    // L matrisini yazdýr
    printf("\nL matrix:\n");
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%10.4f ", L[i][j]);
        }
        printf("\n");
    }

    // U matrisini yazdýr
    printf("\nU matrix:\n");
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%10.4f ", U[i][j]);
        }
        printf("\n");
    }

    // Ýleri yerine koyma (Forward Substitution)
    for (i = 0; i < n; i++) {
        sum = 0.0;
        for (j = 0; j < i; j++) {
            sum = sum + L[i][j] * y[j];
        }
        y[i] = (c[i] - sum) / L[i][i];
    }

    // y vektörünü yazdýr
    printf("\nForward (y values):\n");
    for (i = 0; i < n; i++) {
        printf("y[%d] = %.4f\n", i + 1, y[i]);
    }

    // Geriye yerine koyma (Backward Substitution), U köþegen = 1 olduðundan sadeleþtirilmiþ
    for (i = n - 1; i >= 0; i--) {
        sum = 0.0;
        for (j = i + 1; j < n; j++) {
            sum = sum + U[i][j] * x[j];
        }
        x[i] = y[i] - sum;
    }

    // x çözüm vektörünü yazdýr
    printf("\nBackward (solution x):\n");
    for (i = 0; i < n; i++) {
        printf("x[%d] = %.4f\n", i + 1, x[i]);
    }

    // Bellek temizliði
    free_matrix(A, n);
    free_matrix(L, n);
    free_matrix(U, n);
    free(c);
    free(y);
    free(x);
}

void gauss_seidel() {
    int n, max_iter, i, j, k, iter, max_row, converged = 0;
    double tol, sum, x_new, dx, max_val;

    printf("Enter the number of variables (system size n): ");
    scanf("%d", &n);

    double **A = alloc_matrix(n);
    double *c = malloc(n * sizeof(double));
    double *x = malloc(n * sizeof(double));
    double *oldX = malloc(n * sizeof(double));

    printf("\nNow enter the coefficient matrix A (%dx%d):\n", n, n);
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("  Enter A[%d][%d]: ", i, j);
            scanf("%lf", &A[i][j]);
        }
    }

    printf("\nNow enter the constants vector c:\n");
    for (i = 0; i < n; i++) {
        printf(" You're entering c[%d]: ", i);
        scanf("%lf", &c[i]);
    }

    printf("\nNow enter initial guesses for the unknowns (x1 to x%d):\n", n);
    for (i = 0; i < n; i++) {
        printf("for x%d: ", i + 1);
        scanf("%lf", &x[i]);
    }

    printf("\nEnter the desired tolerance (e.g., 0.001): ");
    scanf("%lf", &tol);
    printf("Note: Iterations will stop early if the maximum change in variables is below the tolerance (%g).\n\n", tol);
    printf("Enter the maximum number of iterations: ");
    scanf("%d", &max_iter);

    // Satýr takasýyla pivotlama (diagonal dominance için)
    for (i = 0; i < n; i++) {
        max_row = i;
        max_val = fabs(A[i][i]);
        for (j = i + 1; j < n; j++) {
            if (fabs(A[j][i]) > max_val) {
                max_val = fabs(A[j][i]);
                max_row = j;
            }
        }
        if (max_row != i) {
            for (k = 0; k < n; k++) {
                double tmp = A[i][k];
                A[i][k] = A[max_row][k];
                A[max_row][k] = tmp;
            }
            double tmpc = c[i];
            c[i] = c[max_row];
            c[max_row] = tmpc;
        }
    }

    // Baþlýk satýrý
    printf("\nIter ");
    for (i = 0; i < n; i++) {
        printf("      x%-2d        |dx%-2d|    ", i+1, i+1);
    }
    printf("\n-----");
    for (i = 0; i < n; i++) {
        printf("----------------------------");
    }
    printf("\n");

    // Ýlk iterasyon çýktýsý
    printf("%5d", 1);
    for (i = 0; i < n; i++) {
        printf(" %12.6f     %-10s", x[i], "-");
    }
    printf("\n");

    for (iter = 2; iter <= max_iter && !converged; iter++) {
        for (i = 0; i < n; i++) oldX[i] = x[i];

        for (i = 0; i < n; i++) {
            sum = c[i];
            for (j = 0; j < n; j++) {
                if (j != i) sum -= A[i][j] * x[j];
            }
            x_new = sum / A[i][i];
            x[i] = x_new;
        }

        printf("%5d", iter);
        int done = 1;
        for (i = 0; i < n; i++) {
            dx = fabs(x[i] - oldX[i]);
            if (dx > tol) done = 0;
            printf(" %12.6f  %12.6f", x[i], dx);
        }
        printf("\n");

        if (done) 
		converged = 1;
    }

    // Final sonuç
    printf("\nFinal Solution (x vector):\n");
    for (i = 0; i < n; i++) {
        printf("  x%d = %10.6f\n", i+1, x[i]);
    }

    free_matrix(A, n);
    free(c);
    free(x);
    free(oldX);
}


void numerical_derivative() {
    char raw[256], func[512];
    double point, h;

    getchar();
    printf("Enter function f(x):\n");
    fgets(raw, sizeof(raw), stdin);
    raw[strcspn(raw, "\n")] = '\0';
    preprocess_log_base(raw, func);

    printf("Point x: ");
    scanf("%lf", &point);

    printf("Step size h (0.00001): ");
    if (scanf("%lf", &h) != 1) h = EPSILON;

    double f   = evaluate(func, point);
    double fp  = evaluate(func, point + h);
    double fm  = evaluate(func, point - h);
    
    
    // Karmaþýk sayý kontrolü
    if (isnan(f) || isinf(f) || isnan(fp) || isinf(fp) || isnan(fm) || isinf(fm)) {
        printf("Complex or undefined value encountered in numerical_derivative, aborting.\n");
        return;
    }

    double df_fwd = (fp - f) / h;
    double df_bwd = (f  - fm) / h;
    double df_ctr = (fp - fm) / (2.0 * h);

    printf("\nNumerical derivative at x = %.6f\n", point);
    printf("  Forward  diff : %.6f\n", df_fwd);
    printf("  Backward diff : %.6f\n", df_bwd);
    printf("  Central  diff : %.6f\n", df_ctr);
}

// integral = ( h/3 ) * [ f0 + 4*f1 + 2*f2 + 4*f3 + ... + fn ]
void simpson_one_third() {
    char raw[256], func[512];
    double a, b;
    int n;
    int i;
    double h;
    double xi;
    double coeff;
    double sum;
    double I;

    getchar();
    printf("Enter function f(x):\n");
    fgets(raw, sizeof(raw), stdin);
    raw[strcspn(raw, "\n")] = '\0';
    preprocess_log_base(raw, func);

    printf("Enter lower bound a: ");
    scanf("%lf", &a);

    printf("Enter upper bound b: ");
    scanf("%lf", &b);

    printf("Sub-interval count n (even): ");
    scanf("%d", &n);

    if (n % 2 != 0) {
        printf("For Simpson 1/3, n must be even.\n");
        return;
    }

    h = (b - a) / n;
    sum = evaluate(func, a) + evaluate(func, b);

    for (i = 1; i < n; i++) {
        xi = a + i * h;
        if (i % 2 == 0) {
            coeff = 2.0;
        } else {
            coeff = 4.0;
        }
        sum = sum + coeff * evaluate(func, xi);
    }

    I = (h / 3.0) * sum;

    printf("\nSimpson 1/3 (n = %d): %.6f\n", n, I);
}


// integral = ( 3*h/8 ) * [ f0 + 3*f1 + 3*f2 + f3 + ... + fn ]
void simpson_three_eighth() {
    char raw[256], func[512];
    double a, b;
    int n;
    int i;
    double h;
    double xi;
    double coeff;
    double sum;
    double I;

    getchar();
    printf("Enter function f(x):\n");
    fgets(raw, sizeof(raw), stdin);
    raw[strcspn(raw, "\n")] = '\0';
    preprocess_log_base(raw, func);

    printf("Enter lower bound a: ");
    scanf("%lf", &a);

    printf("Enter upper bound b: ");
    scanf("%lf", &b);

    printf("Sub-interval count n (multiple of 3): ");
    scanf("%d", &n);

    if (n % 3 != 0) {
        printf("For Simpson 3/8, n must be a multiple of 3.\n");
        return;
    }

    h = (b - a) / n;
    sum = evaluate(func, a) + evaluate(func, b);

    for (i = 1; i < n; i++) {
        xi = a + i * h;
        if (i % 3 == 0) {
            coeff = 2.0;
        } else {
            coeff = 3.0;
        }
        sum = sum + coeff * evaluate(func, xi);
    }

    I = (3.0 * h / 8.0) * sum;

    printf("\nSimpson 3/8 (n = %d): %.6f\n", n, I);
}

// integral = (h/2)*[f0 + 2*f1 + 2*f2 + ... + fn]
void trapezoidal() {
    char raw[256], func[512];
    double a, b;
    int n;

    getchar();
    printf("Enter function f(x):\n");
    fgets(raw, sizeof(raw), stdin);
    raw[strcspn(raw, "\n")] = '\0';
    preprocess_log_base(raw, func);

    printf("Enter lower bound a: ");
    scanf("%lf", &a);
    printf("Enter upper bound b: ");
    scanf("%lf", &b);
    printf("Sub-interval count n: ");
    scanf("%d", &n);

    if (n < 1) {
        printf("n must be >= 1\n");
        return;
    }

    double h   = (b - a) / n;
    double sum = evaluate(func, a) + evaluate(func, b);

    int i;
    for (i = 1; i < n; i++) {
        double xi = a + i * h;
        sum += 2.0 * evaluate(func, xi);
    }

    double I = (h / 2.0) * sum;
    printf("\nTrapezoidal rule (n = %d): %.6f\n", n, I);
}

// F(x) = f0 + (x - x0)/h * delta f0 + (x - x0)*(x - x1)/h^2 * delta^2 f0 / 2! + (x - x0)*(x - x1)*(x - x2)/h^3 * delta^3 f0 / 3! + ...
void gregory_newton_interpolation() {
    int n, i, j, k, should_print;
    double x0, h, xp, p, yp, pterm, coeff;

    printf("Enter number of data points: ");
    scanf("%d", &n);

    if (n < 2) {
        printf("You must enter at least 2 points.\n");
        return;
    }

    double *ys = (double *)malloc(n * sizeof(double));
    double *xs = (double *)malloc(n * sizeof(double));

    printf("Enter starting x (x0): ");
    scanf("%lf", &x0);

    printf("Enter step size h: ");
    scanf("%lf", &h);

    printf("Enter value to interpolate x_p: ");
    scanf("%lf", &xp);

    for (i = 0; i < n; i++) {
        xs[i] = x0 + i * h;
    }

    printf("Enter %d y values:\n", n);
    for (i = 0; i < n; i++) {
        printf("  f(%.6g) = ", xs[i]);
        scanf("%lf", &ys[i]);
    }

    // fark tablosu
    double **diff = (double **)malloc(n * sizeof(double *));
    for (i = 0; i < n; i++) {
        diff[i] = (double *)calloc(n, sizeof(double));
        diff[i][0] = ys[i];
    }

    for (j = 1; j < n; j++) {
        for (i = 0; i < n - j; i++) {
            diff[i][j] = diff[i + 1][j - 1] - diff[i][j - 1];
        }
    }

    // tabloyu yazdýr
    printf("\n%-10s%-12s", "x", "f(x)");
    for (j = 1; j < n; j++) {
        printf("delta^%df(x)     ", j);
    }
    printf("\n");

    for (i = 0; i < n; i++) {
        printf("%-10.6g%-12.6g", xs[i], diff[i][0]);
        for (j = 1; j < n - i; j++) {
            printf("%-12.6g", diff[i][j]);
        }
        printf("\n");
    }

    // interpolasyon
    p = (xp - x0) / h;
    yp = diff[0][0];
    pterm = 1.0;

    for (k = 1; k < n; k++) {
        pterm *= (p - (k - 1));
        yp += (pterm / factorial_int(k)) * diff[0][k];
    }

    printf("\nInterpolated value at x = %.6g : %.6g\n", xp, yp);

    // interpolasyon fonksiyonu gösterimi
    printf("\nInterpolated polynomial in terms of x:\n");
    printf("f(x) roughly = %.6g", diff[0][0]);

    for (k = 1; k < n; k++) {
        coeff = diff[0][k] / factorial_int(k);
        should_print = (coeff != 0);

        if (should_print) {
            printf(" + %.6g", coeff);
            for (j = 0; j < k; j++) {
                printf("*((x - %.6g)/%.6g", x0, h);
                if (j > 0) printf(" - %d", j);
                printf(")");
            }
        }
    }

    printf("\n");

    // bellek serbest býrak
    for (i = 0; i < n; i++) free(diff[i]);
    free(diff);
    free(ys);
    free(xs);
}




