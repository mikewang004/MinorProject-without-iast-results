      Function InvP(I,J,Spread)
      Implicit None

      Include 'commons.inc'

      Integer I,J
      Double Precision InvP,Spread

         If(Langmuir(I,J)) Then         
            InvP = Ki(I,J)/(Dexp(Spread/Nimax(I,J))-1.0d0)         
         Else
            InvP = (Ki(I,J)/(Dexp(Spread*Pow(I,J)/Nimax(I,J))-1.0d0))
     &                **(1.0d0/Pow(I,J))
         Endif
         
      Return
      End
      
