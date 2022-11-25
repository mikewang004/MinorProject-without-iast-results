      Integer Maxcomp,Maxterm

      Parameter (Maxcomp = 10)
      Parameter (Maxterm = 10)

      Integer Ncomp
      
      Double Precision Nimax(Maxcomp,Maxterm),Ki(Maxcomp,Maxterm)
     $     ,Pow(Maxcomp,Maxterm)

      Logical Langmuir(Maxcomp,Maxterm)

      Common /Myiast/ Nimax,Ki,Pow,Ncomp,Langmuir
