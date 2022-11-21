      Function IsothermPure(I,J,P)
      Implicit None

C     Multisite Langmuir Freundlich isotherm
      
      Include 'commons.inc'

      Integer I,J
      Double Precision Dummy,P,Isothermpure
      

      If(Langmuir(I,J)) Then
         Dummy = Ki(I,J)*P
      Else
         Dummy = Ki(I,J)*(P**Pow(I,J))
      Endif
         
      IsothermPure = Nimax(I,J)*Dummy/(1.0d0+Dummy)
      
      Return
      End
